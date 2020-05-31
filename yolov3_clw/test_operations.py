### 测试 datasets.py
# from utils.datasets import VocDataset
# dataset = VocDataset('D:/valid.txt', img_size=416, with_label=False)
# data = dataset[2]
# print('success !')

import torch

a = torch.tensor([1,2,3,4])
b = torch.tensor([1,3,5,7])
c = a-b
d = c*c
print(c)
print(d)


# import cv2
#
# img = cv2.imread('D:/dataset/val/000001.jpg')
# img = img[:, :, ::-1]
# cv2.imwrite('D:/dataset/val/000001_bgr.jpg', img)



import numpy as np
import torch


# anchors = [1,2,3,4,5]
#
# anchors = torch.FloatTensor(anchors)
# anchors = anchors.repeat(50 * 50, 2)
# anchors = anchors.unsqueeze(0)
#
# print('1111')

# grid1 = np.array([0,1])
# grid2 = np.array([2,3,4])
# a, b = np.meshgrid(grid1, grid2)
# print('a=',a)
# print('b=',b)

# # Add the center offsets
# grid_size=10
# grid = np.arange(grid_size)
# a, b = np.meshgrid(grid, grid)
# print('a=',a)
# print('b=',b)
# x_offset = torch.FloatTensor(a).view(-1, 1)
# y_offset = torch.FloatTensor(b).view(-1, 1)
# print('x_offset=',x_offset)
# print('y_offset=',y_offset)



