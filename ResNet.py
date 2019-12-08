import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils
import torch_utils  # clw add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)

class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)



class ResNet(nn.Module):  # 11 million parameters
    def __init__(self, ):
        super(ResNet, self).__init__()

        # 卷积层部分
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv.add_module("resnet_block1", self.resnet_block(64, 64, 2, first_block=True))
        self.conv.add_module("resnet_block2", self.resnet_block(64, 128, 2))
        self.conv.add_module("resnet_block3", self.resnet_block(128, 256, 2))
        self.conv.add_module("resnet_block4", self.resnet_block(256, 512, 2))

        # 全连接层部分
        self.fc = nn.Sequential(
            utils.GlobalAvgPool2d(),
            utils.FlattenLayer(),
            nn.Linear(512, 10)
        )   # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)

    def forward(self, img):
        feature = self.conv(img)
        #output = self.fc(feature.view(img.shape[0], -1))
        output = self.fc(feature)
        return output

    def resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

if __name__ == "__main__":
    net = ResNet()
    print(net)
    torch_utils.model_info(net, report='summary')  # 'full' or 'summary'

    # 查看每个层输出的shape
    # X = torch.rand((1, 1, 224, 224))
    # for name, layer in net.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)  # conv  output shape:	 torch.Size([1, 512, 7, 7])
    #                                               # fc  output shape:	 torch.Size([1, 10])

    # 获取数据，训练模型
    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


'''
1、ResNet结构：
ResNet(
  (conv): Sequential(
    (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (resnet_block1): Sequential(
      (0): Residual(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Residual(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (resnet_block2): Sequential(
      (0): Residual(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Residual(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (resnet_block3): Sequential(
      (0): Residual(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Residual(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (resnet_block4): Sequential(
      (0): Residual(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): Residual(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (fc): Sequential(
    (0): GlobalAvgPool2d()
    (1): FlattenLayer()
    (2): Linear(in_features=512, out_features=10, bias=True)
  )
)
Model Summary: 76 layers, 1.11784e+07 parameters, 1.11784e+07 gradients

2、测试结果：
epoch 1, loss 0.3911, train acc 0.855, test acc 0.899, time 235.5 sec
epoch 2, loss 0.1248, train acc 0.907, test acc 0.893, time 235.1 sec
'''