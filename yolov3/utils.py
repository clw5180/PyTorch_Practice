import torch
from torch import nn

class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # clw note: TODO
        self.activate = nn.LeakyReLU()  # clw note: TODO

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        Y = self.activate(X)
        return Y

class Residual_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv_block1 = Convolutional(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv_block2 = Convolutional(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, X):
        A1 = self.conv_block1(X)
        Y = self.conv_block2(A1)
        return Y + X

class Route(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)  # x shape: (batch_size, nc, nh, nw)，dim=1即在通道方向进行拼接，即变为（batch_size, nc1+nc2, nh, nw)
                                          # 一个例子：A=torch.ones(2,3)，B=2*torch.ones(4,3)，C=torch.cat((A,B),0)即按行这一维度上拼接（竖着拼），
                                          # 个人认为可以理解为，把一个二维矩阵从上往下拍扁，或者往上下两个方向拉伸，
                                          # 即列数不变，行数增加，变成(6, 3)； torch.sum同理，dim=0表示按行这一维度（竖直方向）相加；
        return out