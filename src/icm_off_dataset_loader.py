import json
import pickle
import clip
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm 

from utils import CaptionDataset
from attribute_evaluator import encode_text

# Run this file to first save all the dataset to disk for offline loading
if __name__ == "__main__":
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Visualize the dataset
    dataset_name = "annotations/captions_train2017"
    data_file = json.load(open(dataset_name + ".json", "rb"))
    train_data = CaptionDataset(data_file, model, device, encode_text, transform=None, train=True)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    print(len(train_dataloader))
    for image, caption_emb, label in tqdm(train_dataloader):
        continue