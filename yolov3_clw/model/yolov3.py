from torch import nn
from .backbone.darknet53 import Darknet53
from .neck.fpn import Fpn


class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_num = 1
        self.out_channels = (self.cls_num + 1 + 4) * 3
        self.backbone = Darknet53()
        self.fpn = Fpn(in_channels=[1024, 512, 256], out_channels=[self.out_channels] * 3)

    def forward(self, X):
        Y_36, Y_61, Y_74 = self.backbone(X)
        Y_105, Y_93, Y_81 = self.fpn(Y_36, Y_61, Y_74)
        return Y_105, Y_93, Y_81

