import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1

from lib_model.backbone_models.resnet import resnet50, resnet18
from lib_model.backbone_models.mobilenet import mobilenet_v2
from lib_model.backbone_models.googlenet import googlenet


feature_extractor_backbone = {
    "net0": nn.Sequential(
        # nn.Conv2d(1, 64 , 17),  # 64@96*96 # for omniglot dataset with input size 64@96*96
        nn.Conv2d(1, 64, 10),  # 64@96*96 # for omniglot dataset with input size 64@105*105
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 64@48*48
        nn.Conv2d(64, 128, 7),
        nn.ReLU(),  # 128@42*42
        nn.MaxPool2d(2),  # 128@21*21
        nn.Conv2d(128, 128, 4),
        nn.ReLU(),  # 128@18*18
        nn.MaxPool2d(2),  # 128@9*9
        # nn.Conv2d(128, 256, 4),
        nn.Conv2d(128, 256, 2),
        nn.ReLU(),  # 256@6*6
    ),
    "net1": nn.Sequential(
        nn.Conv2d(3, 32, 3),  # 32@30*30 # for fc100 dataset with input size 32*32
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 32@15*15
        nn.Conv2d(32, 64, 2),
        nn.ReLU(),  # 64@14*14
        nn.MaxPool2d(2),  # 64@7*7
        nn.Conv2d(64, 128, 2),
        nn.ReLU(),  # 128@6*6
        nn.MaxPool2d(2),  # 128@3*3
        nn.Conv2d(128, 256, 1),
        nn.ReLU(),  # 256@3*3
    ),
    "net2": nn.Sequential(
        nn.Conv2d(3, 64, 19),  # 64@78*78  # for mini imagenet dataset with input size 64@96*96
        # nn.Conv2d(3, 64, 7),  # 64@78*78 # for mini imagenet dataset with input size 64@84*84
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 128@39*39
        nn.Conv2d(64, 128, 4),
        nn.ReLU(),  # 128@36*36
        nn.MaxPool2d(2),  # 256@18*18
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),  # 256@16*16
        nn.MaxPool2d(2),  # 128@8*8
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),  # 256@6*6
    ),
    "net3": nn.Sequential(
        nn.Conv2d(3, 128, 9),  # 64@216*216  # for dataset with input size 64@224*224
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3),  # 128@72*72
        nn.Conv2d(128, 256, 7),
        nn.ReLU(),  # 128@66*66
        nn.MaxPool2d(3),  # 256@22*22
        nn.Conv2d(256, 256, 5),
        nn.ReLU(),  # 256@18*18
        nn.MaxPool2d(2),  # 128@9*9
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),  # 256@7*7
    ),
    "net4": nn.Sequential(
        nn.Conv2d(3, 128, 7),  # 64@106*106  # for dataset with input size 64@112*112
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 128@53*53
        nn.Conv2d(128, 256, 4),
        nn.ReLU(),  # 128@50*50
        nn.MaxPool2d(2),  # 256@25*25
        nn.Conv2d(256, 256, 4),
        nn.ReLU(),  # 256@22*22
        nn.MaxPool2d(2),  # 256@11*11
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),  # 256@9*9
    ),
    "FaceNet": nn.Sequential(
        # *(list(DeepFace.represent(pretrained='vggface2').eval().children()))
        *(list(InceptionResnetV1(pretrained='casia-webface').eval().children())[:-1])
    ),
    # "EfficientNet": nn.Sequential(
    #     *(list(EfficientNet.from_pretrained('efficientnet-b0').children())[:-1])
    # ),
    "ResNet18":
        nn.Sequential(
            *(list(resnet18(pretrained=True).children())[:-1])
        ),
    "ResNet50":
        nn.Sequential(
            *(list(resnet50(pretrained=True).children())[:-1])
        ),
    "MobileNet": nn.Sequential(
        *(list(mobilenet_v2(pretrained=True).children())[:-1])
    ),
    "InceptionV3": nn.Sequential(
        *(list(models.inception_v3(pretrained=True).children())[:-1])  # image size should be N x 3 x 299 x 299
    ),
    "GoogleNet": nn.Sequential(
        *(list(googlenet(pretrained=True).children())[:-1])  # image size should be N x 3 x 224 x 224
    ),
}


class SiameseModel(nn.Module):
    def __init__(self, net_name='net0'):
        super(SiameseModel, self).__init__()
        # different convolutional networks for different datasets
        if net_name in ['net0', 'net2']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        elif net_name == 'net1':
            self.conv = feature_extractor_backbone["net1"]
            self.liner = nn.Sequential(nn.Linear(2304, 4096), nn.Sigmoid())
        elif net_name == 'net3':
            self.conv = feature_extractor_backbone["net3"]
            self.liner = nn.Sequential(nn.Linear(12544, 4096), nn.Sigmoid())
        elif net_name == 'net4':
            self.conv = feature_extractor_backbone["net4"]
            self.liner = nn.Sequential(nn.Linear(20736, 4096), nn.Sigmoid())
        elif net_name in ['EfficientNet', 'InceptionV3']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        elif net_name in ['GoogleNet']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(1024, 4096), nn.Sigmoid())
        elif net_name in ['ResNet18']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(512, 4096), nn.Sigmoid())
        elif net_name in ['ResNet50']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(2048, 4096), nn.Sigmoid())
        elif net_name in ['MobileNet']:
            self.conv = feature_extractor_backbone[net_name]
            # self.liner = nn.Sequential(nn.Linear(62720, 4096), nn.Sigmoid())  # 224*224
            self.liner = nn.Sequential(nn.Linear(11520, 4096), nn.Sigmoid()) # 96*96
            # self.liner = nn.Sequential(nn.Linear(1280, 4096), nn.Sigmoid())  # 32*32
        elif net_name in ['FaceNet']:
            self.conv = feature_extractor_backbone[net_name]
            self.liner = nn.Sequential(nn.Linear(512, 4096), nn.Sigmoid())

        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        # print(summary(self.conv, (3, 96, 96)))
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


# for test
if __name__ == '__main__':
    net = SiameseModel(net_name="ResNet18")
    print(net)
    print(list(net.parameters()))
