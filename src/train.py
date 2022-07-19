import matplotlib.pyplot as plt
import torch
import os
import clip
import numpy as np
from tqdm.auto import tqdm
import json

from utils import CustomDataset, check_dir
from utils.meters import AverageValueMeter
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from attribute_evaluator import get_cached_weights, AttEvaluator, print_metric_table
import torchvision.models
from torch import nn

global_step = 0

import sys, os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def validate(model, device, val_dataloader):
    #  cat_classes = [cat["name"] for cat in sorted(data_file["categories"], key=lambda cat: cat["id"])]
    print("Info: Validating...")
    # get cached weights
    zero_shot_weights, text, att_list = get_cached_weights(model, device)

    # load images
    ground_truth = []
    preds = []
    model.eval()
    print("Info: Computing Logits")
    for images, label in val_dataloader:
        images = images.to(device)
        label = label.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ zero_shot_weights.T
            logits = logits.sigmoid()
            preds.append(logits.cpu().numpy())
            ground_truth.append(label.cpu().numpy().astype(int))

    ground_truth = np.array(ground_truth, dtype=object).squeeze().astype("int")
    preds = np.array(preds, dtype=object).squeeze().astype("float")
    return ground_truth, preds


def validate_resnet(model, device, val_dataloader):
    print("Info: Validating...")

    ground_truth = []
    preds = []
    model.eval()
    print("Info: Computing Logits")
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
    return ground_truth, preds


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def train(model, preprocess, data_file, att_evaluator, device, output_folder, epoch_load=0, isModelSave=False):
    # Load the Training Data
    train_data = CustomDataset(data_file, preprocess, train=True)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # Load validation dataset
    val_data = CustomDataset(data_file, preprocess, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Tensorboard logging
    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    log_dir = check_dir("logs/" + timestamp)
    log = SummaryWriter(log_dir)

    # Mixed Precision Alternative
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # Initialize optimizer and Optimizer
    loss = torch.nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer = torch.optim.AdamW(lr=1e-4, params=model.parameters())

    # Load Saved Model
    continue_training = False
    results_dir = check_dir("results/models")
    epoch = 0
    if len(os.listdir(results_dir)) > 1 and isModelSave:
        # ignore DS store
        checkpoint_path = os.path.join(results_dir, "epoch{}.pth".format(epoch_load))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        continue_training = True

    print("Info: Getting Attribute Embedding")
    text_embedding, text, att_list = get_cached_weights(model, device)

    print("Info: Training...")
    start_epoch = epoch + 1 if continue_training else 0
    end_epoch = 20
    train_loss_meter = AverageValueMeter()
    map_meter = AverageValueMeter()
    global global_step
    for epoch in range(start_epoch, end_epoch):
        print("Epoch: ", epoch)
        for iteration, (images, labels) in enumerate(tqdm(train_dataloader, desc="Training Loop")):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images).to(device)
            norm = torch.norm(image_features, dim=1, keepdim=True).detach()
            image_features /= norm

            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ text_embedding.T
            logits = logits.sigmoid().float()
            labels = labels.float()

            scores_overall, scores_per_class = att_evaluator.evaluate(pred=logits.cpu().detach().numpy().copy(),
                                                                      gt_label=labels.cpu().detach().numpy().copy())
            map_meter.add(scores_per_class['all']['ap'])

            logits = logits.flatten()
            labels = labels.flatten()
            mask = labels < 2
            if mask.sum() != 0:
                logits = logits[mask]
                labels = labels[mask]

            # print("Logits: ", logits)
            # print("Labels: ", labels)

            total_loss = loss(logits, labels)
            print("Loss: ", total_loss.item())
            # make_dot(total_loss).render("clip_torchviz", format="png")
            total_loss.backward()

            train_loss_meter.add(total_loss.item())
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

            if iteration % 40 == 0:
                log.add_scalar("training_loss", train_loss_meter.mean, global_step)
                log.add_scalar("Mean Average Precision", map_meter.mean, global_step)
                train_loss_meter.reset()
                map_meter.reset()

                ground_truth, preds = validate(model, device, val_dataloader)
                val_output_folder = check_dir(output_folder + "/epoch{}".format(epoch))
                val_output_path = os.path.join(val_output_folder, "epoch{}".format(epoch))
                att_evaluator.print_evaluation(
                    pred=preds.copy(),
                    gt_label=ground_truth.copy(),
                    output_file=val_output_path,
                )
                with torch.no_grad():
                    preds = preds.flatten()
                    ground_truth = ground_truth.flatten()
                    mask = ground_truth < 2
                    if mask.sum() != 0:
                        preds = preds[mask]
                        ground_truth = ground_truth[mask]
                    validation_loss = loss(torch.from_numpy(preds).float(), torch.from_numpy(ground_truth).float())
                    log.add_scalar("Validation Loss", validation_loss.item(), global_step)

            global_step += 1

        # Save the model every 5 epochs
        if epoch % 5 == 0 and isModelSave:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(results_dir, "epoch{}.pth".format(epoch))
            )
    return model


def train_resnet(model, data_file, att_evaluator, device, output_folder, epoch_load=0, isModelSave=False):
    # Load the Training Data
    train_data = CustomDataset(data_file, transform=None, train=True)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # Load the validation dataset
    val_data = CustomDataset(data_file, transform=None, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Tensorboard logging
    timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    log_dir = check_dir("logs/" + timestamp)
    log = SummaryWriter(log_dir)

    # Initialize optimizer and Optimizer
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(lr=3e-3, weight_decay=1e-5, params=model.parameters())

    # Load Saved Model
    continue_training = False
    results_dir = check_dir("results/re_models")
    epoch = 0

    if len(os.listdir(results_dir)) > 1 and isModelSave:
        # ignore DS store
        checkpoint_path = os.path.join(results_dir, "epoch{}.pth".format(epoch_load))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        continue_training = True

    print("Info: Getting Attribute Embedding")
    # Run make_caches beforehand

    print("Info: Training...")
    start_epoch = epoch + 1 if continue_training else 0
    end_epoch = 20
    train_loss_meter = AverageValueMeter()
    map_meter = AverageValueMeter()

    global global_step
    for epoch in range(start_epoch, end_epoch):
        print("Epoch: ", epoch)
        for iteration, (images, labels) in enumerate(tqdm(train_dataloader)):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            logits = logits.sigmoid().float()
            labels = labels.float()

            blockPrint()
            results = att_evaluator.print_evaluation(pred=logits.cpu().detach().numpy().copy(),
                                                     gt_label=labels.cpu().detach().numpy().copy())
            map_meter.add(results['PC_t0.5/all/ap'])
            enablePrint()

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

            if iteration % 100 == 0:
                log.add_scalar("training_loss", train_loss_meter.mean, global_step)
                log.add_scalar("Training mAP", map_meter.mean, global_step)
                train_loss_meter.reset()
                map_meter.reset()

                ground_truth, preds = validate_resnet(model, device, val_dataloader)
                blockPrint()
                results = att_evaluator.print_evaluation(pred=preds.copy(),
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

            global_step += 1

        # Save the model every 5 epochs
        if epoch % 5 == 0 and isModelSave:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(results_dir, "epoch{}.pth".format(epoch))
            )
    return model


import torch.nn.functional as F


class ClipModel(nn.Module):
    def __init__(self, backbone_type, path_att_emb):
        super().__init__()
        # device=torch.device("cpu") is important to load weights in float32
        backbone, _ = clip.load(backbone_type, jit=False, device=torch.device("cpu"))
        self.backbone = backbone.visual

        zs_weight = torch.tensor(np.load(path_att_emb), dtype=torch.float32)
        zs_weight = F.normalize(zs_weight, p=2, dim=1)
        self.num_attributes, self.zs_weight_dim = zs_weight.shape

        self.fc = nn.Linear(512, self.zs_weight_dim)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attribute_head = nn.Linear(self.zs_weight_dim, self.num_attributes)
        self.attribute_head.weight.data = zs_weight
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data)
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)


class ResNet50Embedding(nn.Module):
    def __init__(self, pretrained, path_att_emb):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)

        # zs_weight = torch.tensor(np.load(path_att_emb), dtype=torch.float32)
        zs_weight = torch.load(path_att_emb)
        # already normalized
        zs_weight = F.normalize(zs_weight, p=2, dim=1)
        self.num_attributes, self.zs_weight_dim = zs_weight.shape

        self.backbone.fc = nn.Linear(2048, self.zs_weight_dim)
        nn.init.xavier_uniform_(self.backbone.fc.weight)

        self.attribute_head = nn.Linear(self.zs_weight_dim, self.num_attributes)
        self.attribute_head.weight.data = zs_weight
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data)
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.attribute_head(x)
        return x


if __name__ == "__main__":

    # import the json annotations
    dataset_name = "coatt80_val1200"
    data_file = json.load(open(dataset_name + ".json", "rb"))

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

    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    val_data = CustomDataset(data_file, transform=None, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)

    out_dir = "cached weights/"
    zero_shot_out_path = os.path.join(out_dir, "zero_shot_weights.pt")
    model = ResNet50Embedding(pretrained=True, path_att_emb=zero_shot_out_path)

    model = train_resnet(model, data_file=data_file, att_evaluator=evaluator, device=device,
                         output_folder=out_dir,
                         isModelSave=False)

    # Validate Model
    # ground_truth, preds = validate(model, device, val_dataloader)
    ground_truth, preds = validate_resnet(model, device, val_dataloader)

    # Evaluate Predictions
    scores_overall, scores_per_class = evaluator.evaluate(pred=preds.copy(), gt_label=ground_truth.copy())
    print(scores_per_class)
    results = evaluator.print_evaluation(
        pred=preds.copy(),
        gt_label=ground_truth.copy(),
    )

    # print_metric_table(evaluator, results)
