import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None, train=False):
        self.images_map = json_file['images']
        self.json_file = json_file
        self.transform = transform
        self.images = []
        self.labels = []
        self.train = train
        dataset = 'val300'
        if self.train:
            dataset = 'train300'
        for idx in range(len(self.images_map)):
            if self.images_map[idx]['set'] != dataset:
                continue
            self.fillItems(idx)

    def __len__(self):
        return len(self.images)

    def fillItems(self, idx):
        img_path = self.images_map[idx]['file_name']
        image_id = self.images_map[idx]['id']
        image = Image.open(img_path)
        for annotation in self.json_file['annotations']:
            if not self.train and annotation['area'] < 20:
                continue
            if annotation['image_id'] != image_id:
                continue
            x, y, w, h = annotation['bbox']
            left, upper = x, y
            right, lower = x + w, y + h
            self.images.append(image.crop((left, upper, right, lower)))
            label = np.asarray(annotation['att_vec'])
            label[label == -1] = 2
            self.labels.append(label)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

