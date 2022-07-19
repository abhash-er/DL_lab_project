import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None, train=False):
        self.images_map = json_file['images']
        self.json_file = json_file

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                Resize(size=(320, 320), interpolation=BICUBIC),
                _convert_image_to_rgb,
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

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
            if annotation['area'] < 400:
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
            # self.categories.append(self.json_file['categories'][annotation['category_id']])

    def __getitem__(self, idx):
        image = self.images[idx]
        # plt.imshow(image)
        # plt.show()
        # print(self.categories[idx])
        image = self.transform(image)
        # plt.imshow(image.permute((1, 2, 0)))
        # plt.show()
        return image, self.labels[idx]
