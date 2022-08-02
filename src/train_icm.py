import argparse
import json
import os
from datetime import datetime
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from attribute_evaluator import AttEvaluator
from utils.meters import AverageValueMeter
from utils.models import ResNet50Embedding, ClipModel, ResNet50EmbeddingICM, ClipCaptionModel
from utils import check_dir, CustomDataset, CaptionDataset
from attribute_evaluator import encode_text

global_step = 0
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_arguments():
    parser = argparse.ArgumentParser(description="Multi-label-Classification")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--eta-min", type=float, default=3e-5, help="minimum learning rate for the scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="optimizer: weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--batch-size", type=int, default=16, help="minibatch size of labeled training set")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")

    parser.add_argument('--model-name', type=str, default='rn50', help="model name")
    parser.add_argument('--backbone-type', type=str, default='ViT-B/32', help="model name")

    return parser.parse_args()


def train_icm(model, start_batch_idx, train_dataloader, optimizer, loss):
    model.train()
    global global_step
    for batch_idx, (images, caption_embedding, labels) in enumerate((train_dataloader)):
        # labels -> positive/ negative (4 labels)
        optimizer.zero_grad()
        images = images.to(device)
        caption_embedding = caption_embedding.to(device)
        labels = np.array([np.array(label) for label in labels])
        labels = torch.Tensor(labels).to(device)
        # print("Image shape ", images.shape)
        # print("Caption shape ", caption_embedding.shape)
        # print("Lables shape ",labels.shape)
        logits = model(images, caption_embedding)
        # print("Logits shape ", logits.shape)
        logits = logits.sigmoid()
        labels = labels.float()

        # all are 1s and 0s
        logits = logits.flatten()
        labels = labels.flatten()
        
        total_loss = loss(logits, labels)
        total_loss.backward()
        optimizer.step()

    return model


def train(model, images, labels, optimizer, loss, device, train_loss_meter):
    global global_step
    model.train()

    images = images.to(device)
    labels = labels.to(device)  # attribute vector 121

    optimizer.zero_grad()
    logits = model(images)
    logits = logits.sigmoid().float()
    labels = labels.float()

    # blockPrint()
    # results = att_evaluator.print_evaluation(pred=logits.cpu().detach().numpy(),
    #                                          gt_label=labels.cpu().detach().numpy().copy())
    # map_meter.add(results['PC_t0.5/all/ap'])
    # enablePrint()

    logits = logits.flatten()
    labels = labels.flatten()
    mask = labels < 2
    if mask.sum() != 0:
        logits = logits[mask]
        labels = labels[mask]

    total_loss = loss(logits, labels)
    total_loss.backward()

    train_loss_meter.add(total_loss.item())
    optimizer.step()

    global_step += 1

    return model


def validate(model, device, val_dataloader, log, best_val_map, results_dir):
    print("Info: Validating...")
    global global_step

    ground_truth = []
    preds = []
    model.eval()
    for images, label in val_dataloader:
        images = images.to(device)
        label = label.to(device)
        with torch.no_grad():
            logits = model(images)
            logits = logits.sigmoid()

        preds.append(logits.cpu().numpy())
        ground_truth.append(label.cpu().numpy().astype(int))

    ground_truth = np.array(ground_truth, dtype=object).squeeze().astype("int")
    preds = np.array(preds, dtype=object).squeeze().astype("float")

    # Get the evaluation results
    blockPrint()
    results = evaluator.print_evaluation(pred=preds.copy(),
                                         gt_label=ground_truth.copy())
    enablePrint()
    with torch.no_grad():
        preds = preds.flatten()
        ground_truth = ground_truth.flatten()
        mask = ground_truth < 2
        if mask.sum() != 0:
            preds = preds[mask]
            ground_truth = ground_truth[mask]

        validation_loss = loss(torch.from_numpy(preds).float(), torch.from_numpy(ground_truth).float())

    log.add_scalar("Validation Loss", validation_loss.item(), global_step)
    log.add_scalar("Validation mAP", results['PC_t0.5/all/ap'], global_step)

    # save the best model
    if best_val_map < results['PC_t0.5/all/ap']:
        best_val_map = results['PC_t0.5/all/ap']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(results_dir, "model_best.pth")
        )
        print("Saving the best Model at epoch {} with mAP {}, and mAP head {}, mAP medium {} and mAP tail {}".format(
            epoch, best_val_map, results['PC_t0.5/head/ap'],
            results['PC_t0.5/medium/ap'], results['PC_t0.5/tail/ap']))

    return best_val_map


if __name__ == "__main__":
    args = get_arguments()
    # import the json annotations
    dataset_name = "coatt80_val1200"
    data_file = json.load(open(dataset_name + ".json", "rb"))

    caption_dataset_name = "annotations/captions_train2017"
    caption_data_file = json.load(open(caption_dataset_name + ".json", "rb"))
    # Make the dictionaries for evaluator
    attr2idx = {}
    attr_type = {}
    attr_parent_type = {}
    attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}

    for att in data_file["attributes"]:
        attr2idx[att["name"]] = att["id"]

        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])

        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["type"])

        attribute_head_tail[att["freq_set"]].add(att["name"])

    attr_type = {key: list(val) for key, val in attr_type.items()}
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
    attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    out_dir = check_dir("output/" + timestamp)
    run = 0
    output_file = os.path.join(out_dir, "{}_{}.log".format(dataset_name, run))

    # Build evaluator
    evaluator = AttEvaluator(
        attr2idx,
        attr_type=attr_type,
        attr_parent_type=attr_parent_type,
        attr_headtail=attribute_head_tail,
        att_seen_unseen={},
        dataset_name=dataset_name,
        threshold=0.5,
        top_k=8,
        exclude_atts=[],
        output_file=output_file
    )

    # Load Validation Data
    val_data = CustomDataset(data_file, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Load Training Data
    train_data = CustomDataset(data_file, train=True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    cache_dir = "cached weights/"
    zero_shot_out_path = os.path.join(cache_dir, "zero_shot_weights.pt")

    results_dir = "results/" + args.model_name
    # Select the model
    if args.model_name == "rn50":
        model = ResNet50EmbeddingICM(pretrained=True, path_att_emb=zero_shot_out_path, num_captions=10)
    else:
        results_dir = results_dir + "/" + args.backbone_type
        model = ClipCaptionModel(backbone_type=args.backbone_type, path_att_emb=zero_shot_out_path)
    check_dir(results_dir)

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    # TODO Load Caption data
    train_caption_data = CaptionDataset(caption_data_file, model, device, encode_text, transform=None, train=True)
    # The size of train_caption_data is too large, so we will use subset 
    sampled_list = random.sample(range(0,len(train_caption_data)), 500 * 16)
    train_caption_data_subset = torch.utils.data.Subset(train_caption_data, sampled_list)
    train_caption_dataloader = DataLoader(train_caption_data_subset, batch_size=16, shuffle=False,
                                          num_workers=args.num_workers)

    # Tensorboard Logging
    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    log_dir = check_dir("logs/" + timestamp)
    log_tensorboard = SummaryWriter(log_dir)

    # Initialize loss and optimizer
    loss = torch.nn.BCELoss().to(device)
    loss_icm = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=100,
                                                              eta_min=args.eta_min)

    train_loss_meter = AverageValueMeter()
    best_val_map = 0

    for epoch in range(args.num_epochs):
        start_batch = 0
        print("Epoch ", epoch)
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            if batch_idx % 180 == 0:
                # train image caption whenever index hits 180 (2 times)
                model = train_icm(model, start_batch, train_caption_dataloader, optimizer, loss_icm)

            model = train(model, images, labels, optimizer, loss, device, train_loss_meter)

            if batch_idx % 50 == 0:
                log_tensorboard.add_scalar("training_loss", train_loss_meter.mean, global_step)
                # log.add_scalar("Training mAP", map_meter.mean, global_step)
                train_loss_meter.reset()
                # map_meter.reset()
        lr_scheduler.step()

        # Validate After Every 5 Epoch (to save time)
        if epoch % 5 == 0:
            best_val_map = validate(model, device, val_dataloader, log_tensorboard, best_val_map, results_dir)
