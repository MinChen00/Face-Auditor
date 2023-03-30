# Based on https://github.com/floodsung/LearningToCompare_FSL

import torch.nn as nn
import torch.nn.functional as F
import torch
from facenet_pytorch import InceptionResnetV1

from lib_model.backbone_models.resnet import resnet50, resnet18
from lib_model.backbone_models.mobilenet import mobilenet_v2
from lib_model.backbone_models.googlenet import googlenet


class CNNEncoder(nn.Module):
    """docstring for CNNEncoder"""
    def __init__(self, in_channels=3):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out  # 64

in_channels=3
hidden_size=64
out_channels=64

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

feature_extractor_backbone = {
    "SCNN": 
        nn.Sequential(
            nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)),
            nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)),
            nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()),
            nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        ),
    "FaceNet": 
        nn.Sequential(
            nn.Sequential(
            *(list(InceptionResnetV1(pretrained='casia-webface').eval().children())[:-1])),
            nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        ),

    "ResNet18": 
        nn.Sequential(
            nn.Sequential(
            *(list(resnet18(pretrained=True).children())[:-1])),
            nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=1))
        ),
    "ResNet50": 
        nn.Sequential(
            nn.Sequential(
            *(list(resnet50(pretrained=True).children())[:-1])),
            nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        ),
    "MobileNet": 
        nn.Sequential(
            nn.Sequential(
            *(list(mobilenet_v2(pretrained=True).children())[:-1])),
            nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        ),
    "GoogleNet": 
        nn.Sequential(
            nn.Sequential(
            *(list(googlenet(pretrained=True).children())[:-1])),  # image size should be N x 3 x 224 x 224
            nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        ),
}


class FeatureEncoder(nn.Module):
    """docstring for FeatureEncoder"""
    def __init__(self, feature_extractor, in_channels=3, hidden_size=64, out_channels=64):
        super(FeatureEncoder, self).__init__()
        self.encoder = feature_extractor_backbone[feature_extractor]

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*5*5, hidden_size) # image_size: value. (32: 64*1*1) (84: 64*3*3) (96: 64*5*5) (112: 64*6*6) (160: 64*9*9) (224: 64*13*13)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
