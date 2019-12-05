import time
import torch
from torch import nn, optim
import torch.nn.functional as F

import utils
import torch_utils  # clw add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(self.conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

    def conv_block(self, in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                            nn.ReLU(),
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return blk


class DenseNet(nn.Module):  # 0.76 million parameters
    def __init__(self):
        super(DenseNet, self).__init__()

        # 卷积层部分
        self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
        num_convs_in_dense_blocks = [4, 4, 4, 4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            self.conv.add_module("DenseBlosk_%d" % i, DB)
            # 上一个稠密块的输出通道数
            num_channels = DB.out_channels
            # 在稠密块之间加入通道数减半的过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                self.conv.add_module("transition_block_%d" % i, self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.conv.add_module("BN", nn.BatchNorm2d(num_channels))
        self.conv.add_module("relu", nn.ReLU())

        self.fc = nn.Sequential(
            utils.GlobalAvgPool2d(),
            utils.FlattenLayer(),
            nn.Linear(num_channels, 10)
        ) # GlobalAvgPool2d的输出: (Batch, num_channels, 1, 1)

    def forward(self, img):
        feature = self.conv(img)
        # output = self.fc(feature.view(img.shape[0], -1))
        output = self.fc(feature)
        return output

    def transition_block(self, in_channels, out_channels):
        blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
        return blk

if __name__ == "__main__":
    net = DenseNet()
    print(net)
    torch_utils.model_info(net, report='summary')  # 'full' or 'summary'

    # 查看每个层输出的shape
    # X = torch.rand((1, 1, 96, 96))
    # for name, layer in net.named_children():
    #     X = layer(X)
    #     print(name, ' output shape:\t', X.shape)

    # 获取数据，训练模型
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

'''
1、DenseNet结构：
DenseNet(
  (conv): Sequential(
    (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (DenseBlosk_0): DenseBlock(
      (net): ModuleList(
        (0): Sequential(
          (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): Sequential(
          (0): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): Sequential(
          (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): Sequential(
          (0): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (transition_block_0): Sequential(
      (0): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): ReLU()
      (2): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (DenseBlosk_1): DenseBlock(
      (net): ModuleList(
        (0): Sequential(
          (0): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): Sequential(
          (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): Sequential(
          (0): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): Sequential(
          (0): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(192, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (transition_block_1): Sequential(
      (0): BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): ReLU()
      (2): Conv2d(224, 112, kernel_size=(1, 1), stride=(1, 1))
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (DenseBlosk_2): DenseBlock(
      (net): ModuleList(
        (0): Sequential(
          (0): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(112, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): Sequential(
          (0): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(144, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): Sequential(
          (0): BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(176, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): Sequential(
          (0): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(208, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (transition_block_2): Sequential(
      (0): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): ReLU()
      (2): Conv2d(240, 120, kernel_size=(1, 1), stride=(1, 1))
      (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
    (DenseBlosk_3): DenseBlock(
      (net): ModuleList(
        (0): Sequential(
          (0): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(120, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): Sequential(
          (0): BatchNorm2d(152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(152, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (2): Sequential(
          (0): BatchNorm2d(184, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(184, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (3): Sequential(
          (0): BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
          (2): Conv2d(216, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (BN): BatchNorm2d(248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU()
  )
  (fc): Sequential(
    (0): GlobalAvgPool2d()
    (1): FlattenLayer()
    (2): Linear(in_features=248, out_features=10, bias=True)
  )
)
Model Summary: 84 layers, 758226 parameters, 758226 gradients

2、测试结果：
epoch 1, loss 0.4195, train acc 0.848, test acc 0.866, time 186.6 sec
'''