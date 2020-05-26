# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

#from .transforms import RandomErasing   # TODO

# 自注：torchvision毕竟只是torch的一个附属的小package,所以功能还没有那么齐全,
# 目前只能处理类似图像分类这种对图像进行数据增强后无需对标签进行修改的情况,
# 如果类似图像检测这种在做数据增强的时候需要同时修改图像和对应的标签的话,
# 目前是不支持的,所以建议还是自己手写一个dataset,在数据采样的时候对图像
# 和标签进行同步的修改
# https://www.zhihu.com/question/300870083/answer/524735088

def build_transforms(img_size, is_train=True):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            ###T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            ###T.Pad(cfg.INPUT.PADDING),
            ###T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
