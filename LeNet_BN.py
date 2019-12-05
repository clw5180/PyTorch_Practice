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

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = self.batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

    def batch_norm(self, is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 判断当前模式是训练模式还是预测模式
        if not is_training:
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
                # X的形状以便后面可以做广播运算
                mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            # 训练模式下用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 拉伸和偏移
        return Y, moving_mean, moving_var


class LeNet(nn.Module):  # 0.04 million parameters
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
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
epoch 1, loss 1.3572, train acc 0.538, test acc 0.680, time 9.7 sec
epoch 2, loss 0.3593, train acc 0.731, test acc 0.746, time 8.4 sec
epoch 3, loss 0.1962, train acc 0.775, test acc 0.780, time 8.2 sec
epoch 4, loss 0.1313, train acc 0.799, test acc 0.808, time 9.4 sec
epoch 5, loss 0.0974, train acc 0.819, test acc 0.818, time 8.0 sec

增加BN层效果：
epoch 1, loss 0.5694, train acc 0.815, test acc 0.858, time 11.5 sec
epoch 2, loss 0.1648, train acc 0.880, test acc 0.878, time 10.5 sec
epoch 3, loss 0.0944, train acc 0.897, test acc 0.876, time 11.2 sec
epoch 4, loss 0.0648, train acc 0.904, test acc 0.887, time 10.6 sec
epoch 5, loss 0.0480, train acc 0.912, test acc 0.874, time 10.6 sec
'''