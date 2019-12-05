# 在fashionMNIST数据集上进行测试
import os
import time
import torch
from torch import nn, optim
import utils
import torch_utils  # clw add

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)

class LeNet(nn.Module):  # 0.04 million parameters
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(120, 84),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == "__main__":
    # 查看LeNet每个层的shape
    net = LeNet()
    print(net)
    torch_utils.model_info(net, report='summary')  # 'full' or 'summary'

    # 获取数据，训练模型
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    utils.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    print('clw: training process end!')


'''
1、LeNet结构
LeNet(
  (conv): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): Sigmoid()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): Sigmoid()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc): Sequential(
    (0): Linear(in_features=256, out_features=120, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): Sigmoid()
    (4): Linear(in_features=84, out_features=10, bias=True)
  )
)
Model Summary: 10 layers, 44426 parameters, 44426 gradients


2、对比不同的激活函数
sigmoid测试结果：
epoch 1, loss 1.8063, train acc 0.340, test acc 0.590, time 12.4 sec
epoch 2, loss 0.4717, train acc 0.636, test acc 0.690, time 10.8 sec
epoch 3, loss 0.2560, train acc 0.715, test acc 0.730, time 10.8 sec
epoch 4, loss 0.1714, train acc 0.741, test acc 0.752, time 11.1 sec
epoch 5, loss 0.1257, train acc 0.758, test acc 0.763, time 11.0 sec

ReLU测试结果：
epoch 1, loss 0.9694, train acc 0.641, test acc 0.752, time 8.9 sec
epoch 2, loss 0.2862, train acc 0.780, test acc 0.787, time 7.5 sec
epoch 3, loss 0.1662, train acc 0.815, test acc 0.822, time 7.7 sec
epoch 4, loss 0.1123, train acc 0.837, test acc 0.837, time 7.6 sec
epoch 5, loss 0.0828, train acc 0.850, test acc 0.853, time 7.5 sec

'''