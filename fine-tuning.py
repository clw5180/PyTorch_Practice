import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import utils
import torch_utils

torch_utils.init_seeds()

SHOW_IMG = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'C:/Users/62349/Desktop'  # clw note：数据集从网上下载：https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
if SHOW_IMG:
    utils.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

pretrained_net = models.resnet18(pretrained=True)   # clw note：会自动下载与训练模型，保存在路径
                                                    # C:\Users\62349/.cache\torch\checkpoints\resnet50-19c8e357.pth下
#pretrained_net = models.resnet50(pretrained=True)  # 对于resnet50，batch_size和resnet18相比大概要调小4倍，我的2G显存笔记本调到8才可以
                                                    # 对于resnet18，batch_size由32减小到8后，test acc显著下降；
                                                    # 同样是batch_size=8, resnet50的test acc比resnet18显著提升；
print(pretrained_net.fc)

pretrained_net.fc = nn.Linear(512, 2)
#pretrained_net.fc = nn.Linear(2048, 4)  # clw note: for resnet50
print(pretrained_net.fc)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                            #  fc 中的随机初始化参数和backbone不同，一般需要更大的学习率从头训练。PyTorch可以方便的对模型的
                            #  不同部分设置不同的学习参数，我们在下面代码中将 fc 的学习率设为已经预训练过的部分的10倍；
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

# 微调模型
def train_fine_tuning(net, optimizer, batch_size=32, num_epochs=5):
# def train_fine_tuning(net, optimizer, batch_size=8, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)
'''
epoch 1, loss 1.6939, train acc 0.807, test acc 0.921, time 45.3 sec
epoch 2, loss 0.1885, train acc 0.909, test acc 0.835, time 42.8 sec
epoch 3, loss 0.1997, train acc 0.880, test acc 0.887, time 42.9 sec
epoch 4, loss 0.0496, train acc 0.933, test acc 0.941, time 42.7 sec
epoch 5, loss 0.0422, train acc 0.922, test acc 0.896, time 42.9 sec
'''


###
# clw note：下面对比训练和测试时的数据增强去掉标准化normalize一项的结果，可以看到结果没有明显变化
###
'''
epoch 1, loss 1.9887, train acc 0.802, test acc 0.901, time 43.2 sec
epoch 2, loss 0.2432, train acc 0.881, test acc 0.800, time 42.2 sec
epoch 3, loss 0.0776, train acc 0.915, test acc 0.939, time 42.1 sec
epoch 4, loss 0.0636, train acc 0.913, test acc 0.927, time 42.0 sec
epoch 5, loss 0.0325, train acc 0.941, test acc 0.935, time 42.0 sec
'''

###
# clw note:下面继续对比使用数据增强前后的效果，很神奇的是再一次没用增强反而acc更高, loss更低......
###
no_augs = transforms.Compose([
        transforms.Resize(size=[256, 256]),
        transforms.ToTensor(),
        normalize
    ])

def train_fine_tuning_no_aug(net, optimizer, batch_size=32, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=no_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=no_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    utils.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# train_fine_tuning_no_aug(pretrained_net, optimizer)
'''
epoch 1, loss 1.5915, train acc 0.842, test acc 0.873, time 47.5 sec
epoch 2, loss 0.0559, train acc 0.973, test acc 0.814, time 46.5 sec
epoch 3, loss 0.0211, train acc 0.978, test acc 0.963, time 46.3 sec
epoch 4, loss 0.0069, train acc 0.989, test acc 0.965, time 46.6 sec
epoch 5, loss 0.0026, train acc 0.995, test acc 0.969, time 46.5 sec
'''






