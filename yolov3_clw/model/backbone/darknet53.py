from torch import nn
from ..model_utils import Convolutional, Residual_block

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()  # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），
                                           # 然后把类 FooChild 的对象转换为类 FooParent 的对象; python3可以直接super()...
        self.conv0 = Convolutional(3, 32, 3, 1)
        self.conv0_s2 = Convolutional(32, 64, 3, 1, stride=2)
        self.darknet_block1 = self.darknet_block(64, 32, 64, 1)
        self.conv1_s2 = Convolutional(64, 128, 3, 1, stride=2)
        self.darknet_block2 = self.darknet_block(128, 64, 128, 2)
        self.conv2_s2 = Convolutional(128, 256, 3, 1, stride=2)
        self.darknet_block3 = self.darknet_block(256, 128, 256, 8)
        self.conv3_s2 = Convolutional(256, 512, 3, 1, stride=2)
        self.darknet_block4 = self.darknet_block(512, 256, 512, 8)
        self.conv4_s2 = Convolutional(512, 1024, 3, 1, stride=2)
        self.darknet_block5 = self.darknet_block(1024, 512, 1024, 4)

    def forward(self, X):
        X = self.conv0(X)
        X = self.conv0_s2(X)
        X = self.darknet_block1(X)
        X = self.conv1_s2(X)
        X = self.darknet_block2(X)
        X = self.conv2_s2(X)
        Y_36 = self.darknet_block3(X)  # clw note: 第36层输出的feature map，一路作为下面第37层的conv输入，一路通过route层直接和98层输出相加，
                                      #           因此需要单独把这一层feature map抽出来
        X = self.conv3_s2(Y_36)
        Y_61 = self.darknet_block4(X)  # clw note: 第61层输出的feature map，一路作为下面第62层的conv输入，一路通过route层直接和86层输出相加
        X = self.conv4_s2(Y_61)
        Y_74 = self.darknet_block5(X)
        return Y_36, Y_61, Y_74


    def darknet_block(self, in_channels, mid_channels, out_channels, num_residuals):
        blk = []
        for i in range(num_residuals):
            blk.append(Residual_block(in_channels, mid_channels, out_channels))
        return nn.Sequential(*blk)

