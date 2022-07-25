import json

import clip
import torch
from torch.utils.data import DataLoader

from attribute_evaluator import get_cached_weights
from utils import CustomDataset

if __name__ == "__main__":
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    text_embedding, text, att_list = get_cached_weights(model, device)
    # print("Text Embedding", text_embedding.shape)
    # print("Text", text)
    # print("Attribute List", att_list.shape)

    # Visualize the dataset
    dataset_name = "coatt80_val1200"
    data_file = json.load(open(dataset_name + ".json", "rb"))
    train_data = CustomDataset(data_file, transform=None, train=True)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    # for image, label in train_dataloader:
    #     continue
