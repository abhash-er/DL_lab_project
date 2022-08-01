import clip
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision

class ClipModel(nn.Module):
    def __init__(self, backbone_type, path_att_emb):
        super().__init__()
        # device=torch.device("cpu") is important to load weights in float32
        backbone, _ = clip.load(backbone_type, jit=False, device=torch.device("cpu"))
        self.backbone = backbone.visual
        self.transform = _

        zs_weight = torch.load(path_att_emb)
        zs_weight = F.normalize(zs_weight, p=2, dim=1)
        self.num_attributes, self.zs_weight_dim = zs_weight.shape

        if "RN" in backbone_type:
            self.fc = nn.Linear(1024, self.zs_weight_dim)
        else:
            self.fc = nn.Linear(512, self.zs_weight_dim)

        self.attribute_head = nn.Linear(self.zs_weight_dim, self.num_attributes)
        self.attribute_head.weight.data = zs_weight.float()
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data).float()
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.attribute_head(x)
        return x


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
        self.attribute_head.weight.data = zs_weight.float()
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data).float()
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.attribute_head(x)
        return x

class ResNet50EmbeddingICM(nn.Module):
    def __init__(self, pretrained, path_att_emb, num_captions=10):
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
        self.attribute_head.weight.data = zs_weight.float()
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data).float()
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x, y=None):
        if y is not None:
            x = self.backbone(x)
            y.requires_grad = False
            y = y.float()
            x = x.unsqueeze(1).bmm(y.permute(0, 2, 1)).squeeze(1)
            return x.T
        else:
            x = self.backbone(x)
            x = self.attribute_head(x)
            return x

class ClipCaptionModel(nn.Module):
    def __init__(self, backbone_type, path_att_emb, num_captions=10):
        super().__init__()
        # device=torch.device("cpu") is important to load weights in float32
        backbone, _ = clip.load(backbone_type, jit=False, device=torch.device("cpu"))
        self.backbone = backbone.visual

        zs_weight = torch.load(path_att_emb)
        zs_weight = F.normalize(zs_weight, p=2, dim=1)
        self.num_attributes, self.zs_weight_dim = zs_weight.shape

        if "RN" in backbone_type:
            self.fc = nn.Linear(1024, self.zs_weight_dim)
            self.fc_caption = nn.Linear(1024, self.zs_weight_dim)
        else:
            self.fc = nn.Linear(512, self.zs_weight_dim)
            self.fc_caption = nn.Linear(512, self.zs_weight_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.fc_caption.weight)

        self.attribute_head = nn.Linear(self.zs_weight_dim, self.num_attributes)
        self.attribute_head.weight.data = zs_weight.float()
        self.attribute_head.bias.data = torch.zeros_like(self.attribute_head.bias.data).float()
        self.attribute_head.weight.requires_grad = False
        self.attribute_head.bias.requires_grad = False

    def forward(self, x, y=None):
        if y is not None:
            x = self.backbone(x)
            x = self.fc_caption(x)
            y.requires_grad = False
            y = y.float()
            x = x.unsqueeze(1).bmm(y.permute(0, 2, 1)).squeeze(1)
            return x.T
        else:
            x = self.backbone(x)
            x = self.fc(x)
            x = self.attribute_head(x)
            return x