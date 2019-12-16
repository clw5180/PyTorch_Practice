import os
import time
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utils
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

SHOW_IMG = False

if SHOW_IMG:
    #utils.set_figsize()
    img = Image.open('./data/cat1.jpg')
    plt.imshow(img)
    plt.show()

    # 大部分图像增广方法都有一定的随机性。为了方便观察图像增⼴的效果，接下来我们定义一个辅助函数 apply 。
    # 这个函数对输入图像 img 多次运行图像增广方法 aug 并展示所有的结果。
    def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
        Y = [aug(img) for _ in range(num_rows * num_cols)]
        utils.show_images(Y, num_rows, num_cols, scale)

    # 水平翻转（均为⼀半概率）
    apply(img, torchvision.transforms.RandomHorizontalFlip())

    # 垂直翻转
    apply(img, torchvision.transforms.RandomVerticalFlip())

    # 随机裁剪（面积在10%~100%，宽高比在0.5~2随机选取，最终resize为200x200）
    apply(img, torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2)))

    # 变化亮度、色调、对比度或（亮度在50%~150%随机调整）
    apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
    apply(img, torchvision.transforms.ColorJitter(hue=0.5))
    apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
    # 同时改变亮度、色调、饱和度和对比度
    apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))

    # 叠加多种增强的方法
    shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)


# 使用数据增强的方法训练模型
all_imges = torchvision.datasets.CIFAR10(train=True, root="~/Datasets/CIFAR", download=True)  # all_imges的每一个元素都是(image, label)
if SHOW_IMG:
    utils.show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8);

flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])  # 使用 ToTensor 将小批量图像转成PyTorch需要的格式，即形状为(批量⼤小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数

no_aug = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4

def load_cifar10(is_train, augs, batch_size, root="~/Datasets/CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, utils.resnet18(10)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=5)


# clw note:下面对比使用数据增强前后的效果，很神奇的是没用增强反而acc更高, loss更低......
#          可能原因是：第一，只是线上而不是线下增强；第二，epoch训的不多，还看不出效果；

#train_with_data_aug(no_aug, no_aug)
'''
epoch 1, loss 1.4010, train acc 0.494, test acc 0.423, time 86.4 sec
epoch 2, loss 0.4951, train acc 0.648, test acc 0.609, time 85.4 sec
epoch 3, loss 0.2717, train acc 0.715, test acc 0.586, time 85.6 sec
epoch 4, loss 0.1739, train acc 0.756, test acc 0.681, time 85.8 sec
epoch 5, loss 0.1190, train acc 0.791, test acc 0.714, time 85.4 sec  
'''

train_with_data_aug(flip_aug, no_aug)
'''
epoch 1, loss 1.3929, train acc 0.496, test acc 0.570, time 86.6 sec
epoch 2, loss 0.5131, train acc 0.637, test acc 0.573, time 85.6 sec
epoch 3, loss 0.2880, train acc 0.696, test acc 0.571, time 85.6 sec
epoch 4, loss 0.1903, train acc 0.734, test acc 0.660, time 85.6 sec
epoch 5, loss 0.1366, train acc 0.761, test acc 0.708, time 85.6 sec
'''


