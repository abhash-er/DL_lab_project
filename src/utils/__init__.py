import os
import pickle
import random
from tqdm.auto import tqdm 

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip


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
                Resize((224, 224), interpolation=BICUBIC),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # CenterCrop(224),
                RandomHorizontalFlip(),
                _convert_image_to_rgb,
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


class CaptionDataset(Dataset):
    def __init__(self, json_file, model, device, encode_text_fun, transform=None, train=True):
        self.json_file = json_file
        self.images_map = json_file["images"]
        self.model = model
        self.device = device
        self.root = "datasets/coco/train2017/"
        self.encode_text = encode_text_fun

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                Resize((224, 224), interpolation=BICUBIC),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # CenterCrop(224),
                RandomHorizontalFlip(),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.caption_embeddings = []
        self.labels = []
        check_dir("datasets/captions")
        if not os.path.exists("datasets/captions/caption_embeddings.pkl"):
            for idx in range(len(self.images_map)):
                self.fill_items(idx)

            with open("datasets/captions/caption_embeddings.pkl", 'wb') as t:
                pickle.dump(self.caption_embeddings, t)

            with open("datasets/captions/labels.pkl", 'wb') as t:
                pickle.dump(self.labels, t)
        else:
            with open("datasets/captions/caption_embeddings.pkl", 'rb') as f:
                self.caption_embeddings = pickle.load(f)

            with open("datasets/captions/labels.pkl", 'rb') as f:
                self.labels = pickle.load(f)

    def fill_items(self, idx):
        img_id = self.images_map[idx]['id']
        caption_label_dict = {}
        captions = []
        negative_ann = []
        for annotation in self.json_file['annotations']:
            if annotation["image_id"] == img_id:
                caption_label_dict[annotation["caption"]] = 1
                captions.append(annotation["caption"])
            else:
                negative_ann.append(annotation)

        # randomly sample 5 other negative captions
        negative_ann = random.sample(negative_ann, 5)
        for annotation in negative_ann:
            caption_label_dict[annotation["caption"]] = 0
            captions.append(annotation["caption"])

        random.shuffle(captions)
        label_caption = []
        for cap in captions:
            label_caption.append(caption_label_dict[cap])
        self.labels.append(label_caption)
        # print(len(self.labels))
        # print(len(self.labels[0]))

        token_caption, caption_embedding = self.encode_text(self.model, captions, device=self.device)
        # print(caption_embedding.shape)
        self.caption_embeddings.append(caption_embedding)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = "datasets/coco/train2017/" + self.images_map[idx]['file_name']
        img = Image.open(img_path)
        img = self.transform(img)
        assert len(self.labels[idx]) >= 10
        # print(self.labels[idx])
        return img, self.caption_embeddings[idx][:10], self.labels[idx][:10]