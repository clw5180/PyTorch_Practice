from torch import nn
from ..utils import Convolutional, Residual_block

class Darknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Convolutional(3, 32, 3, 1)
        self.conv0_s2 = Convolutional(32, 64, 3, 1, stride=2)
        self.resnet_block1 = self.resnet_block(64, 32, 64, 1)
        self.conv1_s2 = Convolutional(64, 128, 3, 1, stride=2)
        self.resnet_block2 = self.resnet_block(128, 64, 128, 2)
        self.conv2_s2 = Convolutional(128, 256, 3, 1, stride=2)
        self.resnet_block3 = self.resnet_block(256, 128, 256, 8)
        self.conv3_s2 = Convolutional(256, 512, 3, 1, stride=2)
        self.resnet_block4 = self.resnet_block(512, 256, 512, 8)
        self.conv4_s2 = Convolutional(512, 1024, 3, 1, stride=2)
        self.resnet_block5 = self.resnet_block(1024, 512, 1024, 4)

    def forward(self, X):
        X = self.conv0(X)
        X = self.conv0_s2(X)
        X = self.resnet_block1(X)
        X = self.conv1_s2(X)
        X = self.resnet_block2(X)
        X = self.conv2_s2(X)
        Y_36 = self.resnet_block3(X)  # clw note: 第36层输出的feature map，一路作为下面第37层的conv输入，一路通过route层直接和98层输出相加，
                                      #           因此需要单独把这一层feature map抽出来
        X = self.conv3_s2(Y_36)
        Y_61 = self.resnet_block4(X)  # clw note: 第61层输出的feature map，一路作为下面第62层的conv输入，一路通过route层直接和86层输出相加
        X = self.conv4_s2(Y_61)
        Y_74 = self.resnet_block5(X)
        return Y_36, Y_61, Y_74


    def resnet_block(self, in_channels, mid_channels, out_channels, num_residuals):
        blk = []
        for i in range(num_residuals):
            blk.append(Residual_block(in_channels, mid_channels, out_channels))
        return nn.Sequential(*blk)

