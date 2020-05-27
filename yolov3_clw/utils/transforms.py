import numpy as np
import cv2
import torch

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa


def build_transforms(img_size, is_train=False):
    if is_train:
        transform = Compose([
            Resize(img_size),  # clw modify
            ToTensor()              # clw modify
        ])
    else:
        transform = Compose([
            Resize(img_size),
            ToTensor()
            # TODO: 归一化，但是使用预训练模型应该不能加
        ])
    return transform


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, *data):  # 测试集只用传img，训练集需要同时传入img和label；因此 data 是可变长参数
        # 这里data一定是tuple，根据长度判断data的有效性
        if len(data) > 2:
            raise Exception('can not pass more than 2 params!')
        elif len(data) == 1:
            data = data[0]   # 如果是tuple内只含有1个元素，则解除tuple，便于后面迭代 data = t(data) 的时候统一返回 img 即可

        for t in self.transforms:
            data = t(data)
        return data

    def add(self, transform):
        self.transforms.append(transform)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        if not isinstance(data, tuple): # 只是传img
            img = data.astype(np.float32)
            img /= 255.0
            img = img[:, :, ::-1].transpose(2, 0, 1)   # clw note: BGR to RGB, [h,w,c] to [c,h,w]
            img = img.astype(np.float32)
            return torch.from_numpy(img)

        else:  # 既有img，又有label
            image, label = data[0], data[1]
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)

            #return (torch.from_numpy(image),  torch.from_numpy(label))
            return (torch.from_numpy(image), label)



class Resize(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        if isinstance(new_size, int):
            self.new_size = (new_size, new_size)  # 规定为 h，w
        self.interpolation = interpolation

    def __call__(self, data):
        if not isinstance(data, tuple):
            img = cv2.resize(data, self.new_size, interpolation=self.interpolation)
            return img

        else:
            img, label = data[0], data[1]
            #orig_h, orig_w = img.shape[:2]
            #ratio_h = self.new_size[0] / orig_h
            #ratio_w = self.new_size[1] / orig_w  # 原图的框 -> resize后的图的框 ，即 orig -> new 比如从500 reize到416，ratio=0.832
            #label[:, 2] *= ratio_w   # clw note：  x_ctr，比如0.2，那么 img从 512->1024, 这个box的 x_ctr 的相对坐标还是 0.2，因此不用乘以 ratio
            #label[:, 3] *= ratio_h   #
            #label[:, 4] *= ratio_w   # clw note:  w，同样也是相对于整张图的大小，因此resize后的相对坐标也不需要任何变换；
            #label[:, 5] *= ratio_h   #               因此这里的 x_ctr, y_ctr, w, h都是不需要任何处理的......

            img = cv2.resize(img, self.new_size, interpolation=self.interpolation)
            return (img, label)

