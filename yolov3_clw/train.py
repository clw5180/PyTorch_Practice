### 数据集默认使用 voc2007+2012
### 训练情况：

# 1、单卡2080Ti
# （1）img_size=416 bs=32 一个batch用时 0.42s

# 2、多卡2080Ti x 2
# （1）img_size=416 bs=64 一个batch用时 0.56s，大概是 bs=32用时的1.3倍，但是batch总数只有原来的0.5倍，实测 1个epoch用时是单卡的 0.7倍




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
import test
import torch.nn as nn
import torch.distributed as dist  # clw note: TODO


### 超参数
lr0 = 1e-3

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

### 模型、日志保存路径
last_model_path = './weights/last.pt'

###

def train():

    # 0、Initialize parameters( set random seed, get cfg info, )
    cfg = opt.cfg
    weights = opt.weights
    img_size = opt.img_size
    batch_size = opt.batch_size
    total_epochs = opt.epochs
    init_seeds()
    data = parse_data_cfg(opt.data)
    train_txt_path = data['train']
    valid_txt_path = data['valid']
    nc = int(data['classes'])

    # 1、加载模型
    model = Darknet(cfg).to(device)
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


    # 2、设置优化器 和 学习率
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0)
    ###### apex need ######
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)  # clw note: 多卡,在 amp.initialize()之后调用分布式代码 DistributedDataParallel否则报错
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level
    ######
    model.nc = nc

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(total_epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    ### 余弦学习率
    #lf = lambda x: (1 + math.cos(x * math.pi / total_epochs)) / 2
    #scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # 3、加载数据集
    train_dataset = VocDataset(train_txt_path, img_size, with_label=True)
    dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,  # TODO: True
                            num_workers=4,
                            collate_fn=train_dataset.train_collate_fn,
                            pin_memory=True)  # TODO：貌似很重要，否则容易炸显存？


    # 4、训练
    print('Starting training for %g epochs...' % total_epochs)
    nb = len(dataloader)

    mloss = torch.zeros(4).to(device)  # mean losses
    for epoch in range(start_epoch, total_epochs):  # epoch ------------------------------
        model.train()  # 写在这里，是因为在一个epoch结束后，调用test.test()时，会调用 model.eval()

        start = time.time()
        print(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 'time_use'))
        #pbar = tqdm(dataloader, ncols=20)  # 行数参数ncols=10，这个值可以自己调：尽量大到不能引起上下滚动，同时满足美观的需求。
        #for i, (img_tensor, target_tensor, img_path, _) in enumerate(pbar):

        for i, (img_tensor, target_tensor, img_path, _) in enumerate(dataloader):
            batch_start = time.time()
            #print(img_path)
            img_tensor = img_tensor.to(device)
            target_tensor = target_tensor.to(device)
            ### 训练过程主要包括以下几个步骤：

            # (1) 前传
            pred = model(img_tensor)

            # (2) 计算损失
            loss, loss_items = compute_loss(pred, target_tensor, model)
            if not torch.isfinite(loss):
                raise Exception('WARNING: non-finite loss, ending training ', loss_items)

            # (3) 损失：反向传播，求出梯度
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # (4) 优化器：更新参数、梯度清零
            # ni = i + nb * epoch  # number integrated batches (since train start)
            # if ni % accumulate == 0:  # Accumulate gradient for x batches before optimizing
            optimizer.step()
            optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6 + '%10.3gs') % (
                '%g/%g' % (epoch, total_epochs - 1), '%.3gG' % mem, *mloss, len(target_tensor), img_size, time.time()-batch_start)

            #pbar.set_description(s)
            ### for debug ###
            if i % 10 == 0:
                print(s)
            # end batch ------------------------------------------------------------------------------------------------

        print('clw: time use per epoch: %.3fs' % (time.time() - start))

        # Update scheduler
        scheduler.step()

        # compute mAP
        test.test(cfg,
                  'cfg/voc.data',
                  img_size=img_size,
                  conf_thres=0.05,
                  iou_thres=0.5,
                  nms_thres=0.5,
                  src_txt_path=valid_txt_path,
                  dst_path='./output',
                  weights=None,
                  model=model)

        # save model 保存模型
        chkpt = {'epoch': epoch,
                 'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),  # clw note: 多卡
                 'optimizer': optimizer.state_dict()}

        torch.save(chkpt, last_model_path)

    print('end')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    parser.add_argument('--device', default='0,1', help='device id (i.e. 0 or 0,1,2,3)') # 默认单卡
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=448, help='resize to this size square and detect')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    opt = parser.parse_args()

    device = select_device(opt.device)
    if device == 'cpu':
        mixed_precision = False

    train()