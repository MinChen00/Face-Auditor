import torch.nn as nn
import learn2learn as l2l
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1

from lib_model.backbone_models.resnet import resnet50, resnet18
from lib_model.backbone_models.mobilenet import mobilenet_v2
from lib_model.backbone_models.googlenet import googlenet

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

in_channels=3
hidden_size=64
out_channels=3

feature_extractor_backbone = {
    "SCNN": nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        ),
    "FaceNet": nn.Sequential(
        # *(list(DeepFace.represent(pretrained='vggface2').eval().children()))
        *(list(InceptionResnetV1(pretrained='casia-webface').eval().children())[:-1])
        ),
    "ResNet18": nn.Sequential(
            *(list(resnet18(pretrained=True).children())[:-1])
        ),
    "ResNet50": nn.Sequential(
            *(list(resnet50(pretrained=True).children())[:-1])
        ),
    "MobileNet": nn.Sequential(
        *(list(mobilenet_v2(pretrained=True).children())[:-1])
        ),
    "GoogleNet": nn.Sequential(
        *(list(googlenet(pretrained=True).children())[:-1])  # image size should be N x 3 x 224 x 224
        ),
}


class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = feature_extractor_backbone[feature_extractor]

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings.view(inputs.size(0), -1)


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, max_pool=True):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim,
                                                  hidden=hid_dim,
                                                  channels=x_dim,
                                                  max_pool=max_pool)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
