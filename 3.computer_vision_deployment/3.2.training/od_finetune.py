import numpy as np
import torch
import math
from torch import nn, optim
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
import pytorch_lightning as pl
from od_datasets import TomatoDataset


num_classes = 3 + 1 # plus background
model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)

# replace classification layer
in_features = model.backbone.out_channels
num_anchors = model.head.classification_head.num_anchors

model.head.classification_head.num_classes = num_classes

cls_logits = torch.nn.Conv2d(model.backbone.out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
torch.nn.init.normal_(cls_logits.weight, std=0.01)
torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

model.head.classification_head.cls_logits = cls_logits


# formato:      batch_size, n_channels, x, y
x = torch.randn(1, 3, 224, 224)
print(x.shape)
model.eval()
print(model(x))
