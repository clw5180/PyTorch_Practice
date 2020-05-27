from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2
from .transforms import build_transforms
from PIL import Image

'''
读取自己数据的基本流程：
1：制作存储了图像的路径和标签信息的txt9
2：将这些信息转化为list，该list每一个元素对应一个样本
3：通过getitem函数，读取数据标签，并返回。

可以把voc2007数据集拷过来，测一下
'''

class VocDataset(Dataset):  # for training/testing
    '''
    类功能：给定训练集或测试集txt所在路径(该txt包含了训练集每一张图片的路径，一行对应一张图片，如 home/dataset/voc2007/train/cat1.jpg),
    以及图片大小img_size，制作可用于迭代的训练集；
    适用目录结构：cat1.txt放置在和cat1.jpg同一文件夹下，cat1.txt是由当前目录下的cat1.xml通过 xml2txt.py脚本转化而来
    '''
    def __init__(self, txt_path, img_size, is_training):
        # 1、获取所有图片路径，存入 list
        with open(txt_path, 'r') as f:
            self.img_file_paths = [x.replace(os.sep, '/') for x in f.read().splitlines()]
        assert len(self.img_file_paths) > 0, 'No images found in %s !' % txt_path

        # 2、获取所有 txt 路径，存入 list
        self.label_file_paths = []
        for img_file_path in self.img_file_paths:
            txt_file_path = img_file_path[:-4] + '.txt'
            assert os.path.isfile(txt_file_path), 'No label_file %s found, maybe need to exec xml2txt.py first !' % txt_file_path
            self.label_file_paths.append(txt_file_path)   # 注意除了有 .jpg .png可能还有.JPG甚至其他...
        if len(self.label_file_paths) == 0:
            is_training = False
        self.is_training = is_training

        # 3、transforms and data aug，如必须要做的 Resize(), ToTensor()
        self.transforms = build_transforms(img_size, is_training)

    def __len__(self):
        return len(self.img_file_paths)

    def __getitem__(self, index):  # 需要得到 img，labels，img_path，orig_size

        # 1、根据 index 读取相应图片，保存图片信息；如果是训练还需要读入label
        img_path = self.img_file_paths[index]
        img_name = img_path.split('/')[-1]
        if self.is_training:
            label_path = self.label_file_paths[index]
            with open(label_path, 'r') as f:
                x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                nL = len(x)
                if nL:
                    labels = torch.zeros((nL, 6))  # add one column to save batch_idx
                                                   # now labels: [ batch_idx, class_idx, x, y, w, h ]
                    labels[:, 1:] = torch.from_numpy(x) # batch_idx is at the first colume, index 0

        # opencv 读取
        img = cv2.imread(img_path)
        if img is None:
            raise Exception('Read image error: %s not exist !' % img_path)
        orig_h, orig_w = img.shape[:2]
        img0 = img.copy()                  # 拷贝一份作为原始副本，便于返回，因为接下来transforms连续操作会改变img

        # PIL 读取
        # img = Image.open(img_path)  # 注意是 img_pil格式
        # orig_w, orig_h = img_pil.size


        if self.is_training:
            img_tensor, label_tensor = self.transforms(img, labels)  # 对 img 和 label 都要做相应的变换
        else:
            img_tensor = self.transforms(img)  # ToTensor 已经转化为 3x416x416 并且完成归一化

        # 2、根据 index 读取相应标签
        if self.is_training:  # 训练
            return img_tensor, label_tensor, img_path, (orig_h, orig_w)
        else:     # 测试
            return img_tensor, img0, img_name

    @staticmethod
    def train_collate_fn(batch):
        img, label, path, shapes = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes  # TODO：如 batch=4，需要对img进行堆叠
        # img 堆叠后变成[bs, 3, 416, 416] 多了bs一个维度,   label原本是[5, 5]  [1, 5]，concat后变成 [n, 5]

    @staticmethod
    def test_collate_fn(batch):
        img_tensor, img0, img_name = list(zip(*batch))  # transposed
        img_tensor = torch.stack(img_tensor, 0)
        return img_tensor, img0, img_name  # TODO：如 batch=4，需要对img和label进行堆叠