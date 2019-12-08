import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

utils.set_figsize()
img = Image.open('../../img/cat1.jpg')
d2l.plt.imshow(img)