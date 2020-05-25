# Some general things or methods of different parts of model
import numpy as np
import torch
from torch import nn

## 1、general module of layers
class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # clw note: TODO
        self.activate = nn.LeakyReLU()  # clw note: TODO, 常用的是0.1

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        Y = self.activate(X)
        return Y


class Residual_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv_block1 = Convolutional(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_block2 = Convolutional(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)

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
                                          # 即列数不变，行数增加，变成(6, 3)； torch.sum()同理，dim=0表示按行这一维度（竖直方向）相加；
                                          # clw note: 自创口诀：在哪维拼接，哪维增加（dim=0，相当于在列的方向上拉伸）；
                                          #                    在哪维相加，哪维缩小（dim=0，在列的方向上相加，拍扁）；
        return out

## 2、print model or part of model's info
def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))

## used by yolo_head.py
def predict_transform(prediction, num_classes, anchors, stride, CUDA=True):
    '''
    功能：将1x1卷积回归的结果tx,ty,tw,th，根据坐标变换公式求出真实的bbox位置bx，by，bw，bh；
    input：某一层的predict结果，
    return：某一层映射到输入图片的result，即坐标变换后的prediction结果

    prediction：  [batchsize, (num_cls +4 + 1)*3, nh, hw ], 比如[4, 85*3, 52, 52]，某一层的预测结果
    anchors:  [10,13,  16,30,  33,23]，出自 [10,13,  16,30,  33,23, 30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    stride:  3个yolo层的stride不同，最后面的stride=32，用于检测大物体，因为感受野大；第2个yolo层stride=16，
             第1个yolo层stride=8，feature map最大，用于检测小物体；
    ''' 
    batch_size = prediction.size(0)  # 4
    grid_size = prediction.size(2)   # 52, if input 416 and first layer:
    bbox_attrs = 5 + num_classes     # COCO:85
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # [4, 85*3, 52*52]
    prediction = prediction.transpose(1, 2).contiguous() # TODO: no need contiguous() ?          [4, 52*52, 85*3]
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs) #  [4, 52*52*3, 85]
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]  # 将anchor映射到feature map上

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])  # bx = sigmoid(tx) + cx
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    # 假设集合A={0, 1}，集合B={2, 3, 4}，则两个集合的笛卡尔积为{(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)}。
    # np.meshgrid()最简洁且清晰的解释就是：把两个数组的笛卡尔积内的元素的第一、二个坐标分别放入两个矩阵中。
    # 结果为两个矩阵， out1=[[0, 1],
    #                       [0, 1],
    #                       [0, 1]]
    #                 out2=[[2, 2],
    #                       [3, 3],
    #                       [4, 4]]

    x_offset = torch.FloatTensor(a).view(-1, 1)  # x_offset= tensor([[0.], [1.], [2.]...[51.], [0.], [1.]...) tensor shape:(52*52, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)  # y_offset= tensor([[0.], [0.], [0.]...[0.], [1.], [1.]...) tensor shape:(52*52, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda() # 参考https://zhuanlan.zhihu.com/p/36984201下方回复

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # cat后变成(52*52, 2)，repeat后变成(52*52*1, 2*3)，view后变成(52*52*1*3 ,2),unsqueeze后变成(1, 52*52*3, 2)

    prediction[:, :, :2] += x_y_offset  # [4, 52*52*3, 2] 和 [1, 52*52*3, 2]相加，前面4是batch_size，这里会进行广播

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # transfer anchor size scale to anchor size in feature map
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)  # [1, 3] -> [1*52*52, 3*1] -> [1, 52*52, 3]
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # bw = pw * exp(tw)

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes])) # 将Sigmoid激活应用于类别分数

    prediction[:, :, :4] *= stride  # transfer feature map's anchor size to real image(input size)

    return prediction