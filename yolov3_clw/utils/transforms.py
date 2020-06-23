import numpy as np
import cv2
import torch
from utils.utils import xyxy2xywh
import random

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa


def build_transforms(img_size, is_train=False):
    if is_train:
        transform = Compose([
            Resize(img_size),  # clw modify
            #LetterBox(img_size),
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
            #img = img.transpose(2, 0, 1)    # clw note：img[:, :, ::-1]很有必要 ，否则会产生大量漏检！事实证明使用官方模型detect，如果用rgb翻转后的图片，很多东西就检不出来了！
            img = img[:, :, ::-1].transpose(2, 0, 1)   # clw note: BGR to RGB, [h,w,c] to [c,h,w]
            img = np.ascontiguousarray(img)
            return torch.from_numpy(img)

        else:  # 既有img，又有label
            img, label = data[0], data[1]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)    # TODO: 这句话如果不加，后面torch.from_numpy(img)会报错
            img_tensor = torch.from_numpy(img).float() / 255
            return (img_tensor, label)



class Resize(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        if isinstance(new_size, int):
            self.new_size = (new_size, new_size)  # 规定为 h，w
        self.interpolation = interpolation

    def __call__(self, data):
        if not isinstance(data, tuple):
            img = cv2.resize(data, self.new_size, interpolation=self.interpolation)
            return img  # TODO：注意不能只返回 img，至少还要返回个 ratio，因为 detect.py 需要知道 resize的比例才能准确定位

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
            ### 画图, for debug
            # h, w = img.shape[:2]
            # img_out = img.copy()
            # for box in label:
            #     x_ctr_box = box[2] * w
            #     y_ctr_box = box[3] * h
            #     w_box = box[4] * w
            #     h_box = box[5] * h
            #     img_out = cv2.rectangle(img, (x_ctr_box - w_box/2, y_ctr_box-h_box/2),  # TODO: 如果 box 是tensor 格式，就不需要转int   TODO
            #                             (x_ctr_box+w_box/2, y_ctr_box+h_box/2), color=(0, 0, 255), thickness=2)
            # i = random.randint(1, 100)
            # cv2.imwrite('./resize_img{}.jpg'.format(i), img_out)
            ###
            return (img, label)


class LetterBox(object):
    def __init__(self, new_shape, interp=cv2.INTER_LINEAR):  # cv2.INTER_AREA
        if isinstance(new_shape, int):
            self.new_shape = (new_shape, new_shape)  # 规定为 h，w
        self.interp = interp

    def __call__(self, data):
        if not isinstance(data, tuple):
            img, _, _ = self.letterbox(data, self.new_shape, self.interp)
            return img  # TODO：同 Resize那里

        else:
            img, label = data[0], data[1]    # img: (375, 500, 3)  label: (n, 6)
            ######
            r = self.new_shape[0] / max(img.shape)  # resize image to img_size
            if r < 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA  # LINEAR for training, AREA for testing
                h, w = img.shape[:2]
                # print('clw: interpolation =', interp )
                # print('clw: interpolation =', interp )
                # print('clw: interpolation =', interp )
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)
            # ######
            h, w = img.shape[:2]  # h:375  w:500
            img, ratio, pad = self.letterbox(img, self.new_shape, self.interp)

            labels = label.clone()  # clw note：这里很重要，因为下面这几句互相会覆盖，所以必须深拷贝出来；
            label[:, 2] = ratio[0] * w * (labels[:, 2] - labels[:, 4] / 2) + pad[0]  # pad width
            label[:, 3] = ratio[1] * h * (labels[:, 3] - labels[:, 5] / 2) + pad[1]  # pad height
            label[:, 4] = ratio[0] * w * (labels[:, 2] + labels[:, 4] / 2) + pad[0]
            label[:, 5] = ratio[1] * h * (labels[:, 3] + labels[:, 5] / 2) + pad[1]

            # convert xyxy to xywh
            label[:, 2:6] = xyxy2xywh(label[:, 2:6])

            # Normalize coordinates 0 - 1
            label[:, [3, 5]] /= img.shape[0]  # height
            label[:, [2, 4]] /= img.shape[1]  # width

            return (img, label)


    def letterbox(self, img, new_shape=(416, 416), interp=cv2.INTER_AREA):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = max(new_shape) / max(shape)
        r = min(r, 1.0)   # only scale down, do not scale up (for better test mAP)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
        return img, ratio, (dw, dh)  # img:(416, 416, 3)  ratio:(0.832, 0.832)    dw:0.0   dh:52.0