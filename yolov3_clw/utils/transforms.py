import numpy as np
import cv2
import torch

#  pip install imgaug
import imgaug as ia
from imgaug import augmenters as iaa


def build_transforms(img_size, is_train=False):
    if is_train:
        transform = Compose([
            ResizeImageAndLabels(img_size),
            ToTensorAndLabels()
        ])
    else:
        transform = Compose([
            ResizeImage(img_size),
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

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def add(self, transform):
        self.transforms.append(transform)


# class ToTensor(object):
#     def __init__(self, max_objects=50, is_debug=False):
#         self.max_objects = max_objects
#         self.is_debug = is_debug
#
#     def __call__(self, sample):
#         image, labels = sample['image'], sample['label']
#
#         image = image.astype(np.float32)
#         image /= 255.0
#         image = np.transpose(image, (2, 0, 1))
#         image = image.astype(np.float32)
#
#         filled_labels = np.zeros((self.max_objects, 5), np.float32)
#         filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
#         return {'image': torch.from_numpy(image), 'label': torch.from_numpy(filled_labels)}


class ToTensorAndLabels(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(labels)}


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.astype(np.float32)
        img /= 255.0
        img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR to RGB, [h,w,c] to [c,h,w]
        img = img.astype(np.float32)

        return torch.from_numpy(img)


class KeepAspect(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        image_new = np.pad(image, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = image_new.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w * (label[:, 1] - label[:, 3]/2)
        y1 = h * (label[:, 2] - label[:, 4]/2)
        x2 = w * (label[:, 1] + label[:, 3]/2)
        y2 = h * (label[:, 2] + label[:, 4]/2)
        # Adjust for added padding
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios from coordinates
        label[:, 1] = ((x1 + x2) / 2) / padded_w
        label[:, 2] = ((y1 + y2) / 2) / padded_h
        label[:, 3] *= w / padded_w
        label[:, 4] *= h / padded_h

        return {'image': image_new, 'label': label}

class ResizeImageAndLabels(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        if isinstance(new_size, int):
            self.new_size = (new_size, new_size)  # 规定为 h，w
        self.interpolation = interpolation

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        orig_h, orig_w = img.shape[:2]
        ratio_h = self.new_size[0] / orig_h
        ratio_w = self.new_size[1] / orig_w   # 比如从400 reize到200，ratio=0.5
        label[:, 1] *= ratio_w   # x
        label[:, 2] *= ratio_h   # y
        label[:, 3] *= ratio_w   # w
        label[:, 4] *= ratio_h   # h

        img = cv2.resize(img, self.new_size, interpolation=self.interpolation)
        return {'image': img, 'label': label}


class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        if isinstance(new_size, int):
            self.new_size = (new_size, new_size)  # 规定为 h，w
        self.interpolation = interpolation

    def __call__(self, img):
        img = cv2.resize(img, self.new_size, interpolation=self.interpolation)
        return img


class ImageBaseAug(object):
    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5))),
                # Add gaussian noise to some images.
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                # Add a value of -5 to 5 to each pixel.
                sometimes(iaa.Add((-5, 5), per_channel=0.5)),
                # Change brightness of images (80-120% of original value).
                sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                # Improve or worsen the contrast of images.
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, sample):
        seq_det = self.seq.to_deterministic()
        image, label = sample['image'], sample['label']
        image = seq_det.augment_images([image])[0]
        return {'image': image, 'label': label}
