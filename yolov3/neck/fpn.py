from torch import nn
from torch.nn import Upsample
from ..utils import Convolutional

class Fpn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # yolo_layer1
        self.conv1_0 = nn.Sequential(
            Convolutional(in_channels, 512, 1, 0),
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, 512, 1, 0),
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, 512, 1, 0),
        )  # 输出Y_79，之后上采样然后concat，在yolo_layer2中也要被使用，因此需要单独提出来
        self.conv1_1 = nn.Sequential(
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, out_channels, 1, 0),
        )

        # yolo_layer2
        self.conv2_0 = nn.Sequential(

        )

    def forward(self, Y_36, Y_61, Y_74):
        Y_79 = self.conv1_0(Y_74)
        Y_82 = self.conv1_1(Y_79)  # out1, from yolo_layer1
        Y_91 =