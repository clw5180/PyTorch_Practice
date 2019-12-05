import time
import torch
from torch import nn, optim
import utils
import torch_utils  # clw add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)

# 一种方法定义网络
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(utils.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net

# 另一种方法定义网络
class VGGNet(nn.Module):  # 128 million parameters，如果是ratio=8的小型VGGNet网络，则为 2 million parameters
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
        super(VGGNet, self).__init__()

        # 卷积层部分
        self.conv = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module("vgg_block_" + str(i + 1), self.vgg_block(num_convs, in_channels, out_channels))

        # 全连接层部分
        self.fc = nn.Sequential(
            utils.FlattenLayer(),
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        #output = self.fc(feature.view(img.shape[0], -1))
        output = self.fc(feature)
        return output

    def vgg_block(self, num_convs, in_channels, out_channels):
        blk = []
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*blk)


if __name__ == "__main__":

    # 方案1、常规的VGG网络
    # 查看VGGNet每个层的shape
    # conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    # fc_features = 512 * 7 * 7 # 根据卷积层的输出算出来的
    # fc_hidden_units = 4096 # 任意
    # net = vgg(conv_arch, fc_features, fc_hidden_units)


    # 方案2、较小的VGG网络
    # ratio = 8
    # small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
    #                    (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    # fc_features = 512 * 7 * 7 # 根据卷积层的输出算出来的
    # fc_hidden_units = 4096 # 任意
    # net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    # print(net)

    # 方案3、clw modify：自建VGGNet类，初始化网络
    ratio = 1
    small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    fc_features = 512 * 7 * 7 # 根据卷积层的输出算出来的
    fc_hidden_units = 4096
    net = VGGNet(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    print(net)


    # 打印网络信息
    torch_utils.model_info(net, report='summary')  # 'full' or 'summary'

    # 打印每一层输出维度
    # X = torch.rand(1, 1, 224, 224)
    # # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    # for name, blk in net.named_children():
    #     X = blk(X)
    #     print(name, 'output shape: ', X.shape)

    # 获取数据，训练模型
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


'''
1、VGGNet结构：
VGGNet(
  (conv): Sequential(
    (vgg_block_1): Sequential(
      (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (vgg_block_2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (vgg_block_3): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU()
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (vgg_block_4): Sequential(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU()
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (vgg_block_5): Sequential(
      (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU()
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (fc): Sequential(
    (0): FlattenLayer()
    (1): Linear(in_features=25088, out_features=4096, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=4096, out_features=10, bias=True)
  )
)
Model Summary: 22 layers, 1.28806e+08 parameters, 1.28806e+08 gradients


2、测试结果（ratio=8，即小型的VGG网络）：
epoch 1, loss 0.5692, train acc 0.787, test acc 0.871, time 248.0 sec 

'''
