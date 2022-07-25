import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from attribute_evaluator import AttEvaluator
from utils import check_dir, CustomDataset
from utils.models import ClipModel, ResNet50Embedding

from tqdm.auto import tqdm
from train import validate
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_arguments():
    parser = argparse.ArgumentParser(description="Multi-label-Classification")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--eta-min", type=float, default=5e-5, help="minimum learning rate for the scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="optimizer: weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--batch-size", type=int, default=16, help="minibatch size of labeled training set")
    parser.add_argument("--num-epochs", type=int, default=100, help="number of epochs")

    parser.add_argument('--model-name', type=str, default='rn50', help="model name")
    parser.add_argument('--backbone-type', type=str, default='ViT-B/32', help="model name")

    return parser.parse_args()


def validate(model, device, val_dataloader):
    ground_truth = []
    preds = []
    model.eval()
    for images, label in tqdm(val_dataloader):
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

    results = evaluator.print_evaluation(pred=preds.copy(),
                                         gt_label=ground_truth.copy())

if __name__ == "__main__":
    args = get_arguments()
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

    # Load Validation Data
    val_data = CustomDataset(data_file, train=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    cache_dir = "cached weights/"
    zero_shot_out_path = os.path.join(cache_dir, "zero_shot_weights.pt")

    results_dir = "results/" + args.model_name
    # Select the model
    if args.model_name == "rn50":
        model = ResNet50Embedding(pretrained=True, path_att_emb=zero_shot_out_path)
        checkpoint_path = "results/rn50/model_best.pth"
    else:
        results_dir = results_dir + "/" + args.backbone_type
        model = ClipModel(backbone_type=args.backbone_type, path_att_emb=zero_shot_out_path)
        checkpoint_path = "results/clip_" + args.backbone_type +  "/model_best.pth"
    check_dir(results_dir)

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    # Initialize loss and optmizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    validate(model, device, val_dataloader)
