import argparse
from model.models import Darknet
from utils.utils import select_device

### 混合精度训练 ###
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Waring: No Apex !!! ')  # https://github.com/NVIDIA/apex
    mixed_precision = False        # not installed
if mixed_precision:
    print('Using Apex !!! ')
######


def train():

    # 0、Initialize parameters
    cfg = opt.cfg

    # 1、Initialize model
    model = Darknet(cfg).to(device)
    # print(model)




    pass





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='config file path')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1,2,3)') # 默认单卡
    opt = parser.parse_args()

    device = select_device(opt.device)
    if device == 'cpu':
        mixed_precision = False

    train()