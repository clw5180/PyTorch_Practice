import argparse
from model.models import Darknet
from utils.utils import select_device, init_seeds
from utils.parse_config import parse_data_cfg
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.datasets import VocDataset
from torch.utils.data import DataLoader

### 超参数
lr0 = 1e-4

###



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

    # 0、Initialize parameters( set random seed, get cfg info, )
    cfg = opt.cfg
    weights = opt.weights
    img_size = opt.img_size
    batch_size = opt.batch_size
    total_epochs = opt.epochs
    init_seeds()
    data_dict = parse_data_cfg(opt.data)
    train_txt_path = data_dict['train']
    valid_txt_path = data_dict['valid']
    nc = int(data_dict['classes'])

    # 1、加载模型
    model = Darknet(cfg).to(device).train()
    if weights.endswith('.pt'):      # TODO: .weights权重格式
        model.load_state_dict(torch.load(weights)['model'])  # TODO：map_location=device ？
    # print(model)

    # 2、设置优化器 和 学习率
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0)
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(total_epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    ### 余弦学习率
    #lf = lambda x: (1 + math.cos(x * math.pi / total_epochs)) / 2
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # 3、加载数据集
    train_dataset = VocDataset(train_txt_path, img_size, is_training=True)
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=False,  # TODO: True
                            num_workers=0,
                            collate_fn=train_dataset.train_collate_fn)


    # 4、训练
    print('Starting training for %g epochs...' % total_epochs)
    for epoch in range(start_epoch, total_epochs):  # epoch ------------------------------
        for i, (img_tensor, target_tensor, img_path, _) in enumerate(dataloader):
            print(i)



    print('end')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1,2,3)') # 默认单卡
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='resize to this size square and detect')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    opt = parser.parse_args()

    device = select_device(opt.device)
    if device == 'cpu':
        mixed_precision = False

    train()