import torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock


class KMnistResNet(ResNet):
    """
    Kmnist Resnet-9 Model, inherits from torchvision.models.resnet.ResNet
    """
    def __init__(self):
        super(KMnistResNet, self).__init__(BasicBlock, [1, 1, 1, 1], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)
