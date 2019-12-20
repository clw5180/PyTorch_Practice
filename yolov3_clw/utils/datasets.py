from torch.utils.data import Dataset
import torch
import cv2
import tqdm
import numpy as np
import os
from utils.utils import xyxy2xywh

'''
读取自己数据的基本流程：
1：制作存储了图像的路径和标签信息的txt
2：将这些信息转化为list，该list每一个元素对应一个样本
3：通过getitem函数，读取数据标签，并返回。

可以把voc2007数据集拷过来，测一下
'''

class LoadImagesAndLabels(Dataset):  # for training/testing
    '''
    类功能：给定训练集txt所在路径, 以及图片大小img_size，制作可用于迭代的训练集；
           该txt包含了训练集每一张图片的路径，一行对应一张图片，如 /mfs/home/dataset/voc2007/train/cat1.jpg
           适用目录结构：cat1.txt放置在和cat1.jpg同一文件夹下
    '''
    def __init__(self, path, img_size=416):
        with open(path, 'r') as f:
            img_paths = f.read().splitlines()
            self.img_paths = list(filter(lambda x: len(x) > 0, img_paths))  # 去掉回车的空行，或者写成self.img_paths = [img_path for img_path in img_paths if len(img_path) > 0]

        n = len(self.img_paths)
        assert n > 0, 'N images found in %s' % path
        self.img_size = img_size
        self.label_files = [ x.replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')  for x in self.img_paths]
        # 如 /mfs/home/dataset/voc2007/train/cat1.txt

        # if n < 1000:  # preload all images into memory if possible
        #    self.imgs = [cv2.imread(img_files[i]) for i in tqdm(range(n)), desc='Reading images']    # TODO

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        # if hasattr(self, 'imgs'):
        #    img = self.imgs[index]  # BGR
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'File Not Found ' + img_path
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size) # 将每幅图resize到img_size

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()   # 每一行的内容: class x_center y_center w h 比如 4 0.43 0.36 0.06 0.24，坐标都是归一化过的
            x = np.array([x.split() for x in lines], dtype=np.float32) # x: (box_num, 5)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw  # 因为图像resize了，所以labels中的坐标信息也要相对变化  TODO：理解的不是很透彻
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh
                print(labels)
        # Augment image and labels
        #if self.augment:
        #    img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size

        # TODO
        # if self.augment:
        #     # random left-right flip
        #     lr_flip = True
        #     if lr_flip and random.random() > 0.5:
        #         img = np.fliplr(img)
        #         if nL:
        #             labels[:, 1] = 1 - labels[:, 1]
        #
        #     # random up-down flip
        #     ud_flip = False
        #     if ud_flip and random.random() > 0.5:
        #         img = np.flipud(img)
        #         if nL:
        #             labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))  # clw note: maybe leave index 0 for batch_size dim
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):   # TODO
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


# 测试
if __name__ == '__main__':
    dataset = LoadImagesAndLabels('C:/Users/62349/Desktop/1.txt')
    print('end!')