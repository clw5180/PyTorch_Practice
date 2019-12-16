from PIL import Image
import numpy as np
import math
import torch
print(torch.__version__)

import utils
from matplotlib import pyplot as plt

img = Image.open('./data/catdog.jpg')
w, h = img.size
print("w = %d, h = %d" % (w, h))

# 生成多个anchors
X = torch.Tensor(1, 3, h, w)  # 构造输入数据
sizes=[0.25]
ratios=[1, 2, 0.5]
Y = utils.MultiBoxPrior(X, sizes=sizes, ratios=ratios)  # clw note: 注意这里s表示anchor面积占总面积大小，而不是边长scale；
                                                             #           实际边长为ws*sqrt(r)和hs/sqrt(r)
#Y = utils.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print('Y.shape:', Y.shape)  # Y.shape: torch.Size([1, 1225224, 4]) 中间那个值表示anchor总数量，如果是3个anchor，对于一幅728x561的图结果就是1225224

boxes = Y.reshape((h, w, len(sizes) * len(ratios), 4))  # 这里的3对应下面show_bboxes里面参数的3组s和r，4是坐标数量
print('访问以(250,250)为中心的第一个anchor:', boxes[250, 250, 0, :])  # 这里是归一化之后的值，实际长度还需要乘以矩阵 torch.tensor([w, h, w, h], dtype=torch.float32)
# 第一个size和ratio分别为0.75和1, 则宽高均为0.75 = 0.7184 + 0.0316 = 0.8206 - 0.0706

# 在某个像素位置显示len(sizes) * len(ratios)个anchor
# fig = plt.imshow(img)
# bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
# utils.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, ['s=0.25, r=1', 's=0.25, r=2', 's=0.25, r=0.5'])
# utils.show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])

# 计算交并比、标注anchor的类别
bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

print('iou:', utils.compute_jaccard(anchors, ground_truth[:, 1:]))

# 显示gt
# fig = plt.imshow(img)
# utils.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')

# 显示一些自定义anchor，根据上面给的坐标
# fig = plt.imshow(img)
# utils.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);


## 训练时标注的anchor类别
labels = utils.MultiBoxTarget(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))  # 该函数将背景类别设为0，
            # 并令从零开始的目标类别的整数索引自加1（1为狗， 2为猫）。我们通过 expand_dims函数为锚框和真实边界框添加样本维，
            # 并构造形状为(批量大小, 包括背景的类别个数, 锚框数)的任意预测结果
print('bbox_offset:', labels[0])
print('bbox_mask:', labels[1])  # 由于我们不关心对背景的检测，有关负类的偏移量不应影响目标函数。
                                # 通过按元素乘法，掩码变量中的0可以在计算目标函数之前过滤掉负类的偏移量
print('cls_labels:', labels[2])


## 测试时用到的nms
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = plt.imshow(img)
utils.show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

output = utils.MultiBoxDetection(cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0), anchors.unsqueeze(dim=0), nms_threshold=0.5)
label = []
bbox = []
fig = plt.imshow(img)
for i in output[0].detach().cpu().numpy(): # i: [class, conf, xmin, ymin, xmax, ymax] 其中class=-1，0,1分别为背景、狗、猫；坐标是归一化后的值
    if i[0] == -1:
        continue
    label.append(('dog=', 'cat=')[int(i[0])] + str(i[1]))
    bbox.append(i)
utils.show_bboxes(fig.axes, torch.tensor(bbox)[:, 2:] * bbox_scale, label)
