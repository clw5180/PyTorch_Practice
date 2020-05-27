import argparse
from model.models import Darknet
from utils.utils import select_device, init_seeds
from utils.parse_config import parse_data_cfg
import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils.datasets import VocDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import compute_loss
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 打印出更多的异常细节


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

        ### model.load_state_dict(torch.load(weights)['model']) # 错误原因：没有考虑类别对不上的那一层，也就是yolo_layer前一层
                                                                #          会报错size mismatch for module_list.81.Conv2d.weight: copying a param with shape torch.Size([255, 1024, 1, 1]) from checkpoint, the shape in current model is torch.Size([75, 1024, 1, 1]).
                                                               #           TODO：map_location=device ？
        chkpt = torch.load(weights, map_location=device)
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
            # model.load_state_dict(chkpt['model'])
        except KeyError as e:
            s = "%s is not compatible with %s" % (opt.weights, opt.cfg)
            raise KeyError(s) from e

    # print(model)
    model.nc = nc

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
                            shuffle=True,  # TODO: True
                            num_workers=0,
                            collate_fn=train_dataset.train_collate_fn)


    # 4、训练
    print('Starting training for %g epochs...' % total_epochs)
    nb = len(dataloader)

    for epoch in range(start_epoch, total_epochs):  # epoch ------------------------------
        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(dataloader, ncols=20)  # 行数参数ncols=10，这个值可以自己调：尽量大到不能引起上下滚动，同时满足美观的需求。
        for i, (img_tensor, target_tensor, img_path, _) in enumerate(pbar):
            #print(img_path)
            img_tensor = img_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # (1) Run model
            pred = model(img_tensor)

            # (2) Compute loss
            loss, loss_items = compute_loss(pred, target_tensor, model)

            # (3) Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # (4) Accumulate gradient for x batches before optimizing
            # ni = i + nb * epoch  # number integrated batches (since train start)
            # if ni % accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, total_epochs - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size)
            pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

    print('end')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1,2,3)') # 默认单卡
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=320, help='resize to this size square and detect')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    opt = parser.parse_args()

    device = select_device(opt.device)
    if device == 'cpu':
        mixed_precision = False

    train()