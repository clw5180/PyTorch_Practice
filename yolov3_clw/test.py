from utils.utils import select_device
from model.models import Darknet
from utils.datasets import VocDataset
from utils.utils import non_max_suppression, load_classes, ap_per_class, xywh2xyxy, bbox_iou, write_to_file
from utils.parse_config import parse_data_cfg

import argparse
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
import numpy as np
import torch.nn as nn

SHOW = True
SAVE = False

def test(cfg,
         data,
         batch_size,
         img_size,
         conf_thres,
         iou_thres,
         nms_thres,
         src_txt_path='./valid.txt',
         dst_path='./output',
         weights=None,
         model=None,
         log_file_path='log.txt'):

    # 0、初始化一些参数
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    names = load_classes(data['names'])

    # 1、加载网络
    if model is None:
        device = select_device(opt.device)
        model = Darknet(cfg)
        if weights.endswith('.pt'):      # TODO: .weights权重格式
            model.load_state_dict(torch.load(weights)['model'])  # TODO：map_location=device ？
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # clw note: 多卡
    else:
        device = next(model.parameters()).device  # get model device
    model.to(device).eval()

    # 2、加载数据集
    valid_dataset = VocDataset(src_txt_path, img_size, with_label=True)
    dataloader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,    # TODO
                            collate_fn=valid_dataset.train_collate_fn,   # TODO
                            pin_memory=True)

    # 3、预测，前向传播
    image_nums = 0
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@{}'.format(iou_thres), 'F1')
    #s = ('%20s' + '%10s' * 6) % ('Class', 'ImgNum', 'Target', 'P', 'R', 'mAP@0.5', 'F1')

    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []


    pbar = tqdm(dataloader)
    for i, (img_tensor, target_tensor, img_path, _) in enumerate(pbar):
        start = time.time()
        img_tensor = img_tensor.to(device)   # (bs, 3, 416, 416)
        target_tensor = target_tensor.to(device)

        # Disable gradients
        with torch.no_grad():
            # (1) Run model
            output = model(img_tensor)[0]   # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # (2) NMS
            nms_output = non_max_suppression(output, conf_thres, nms_thres)
            s = 'time use per batch: %.3fs' % (time.time() - start)

        pbar.set_description(s)

        for batch_idx, pred in enumerate(nms_output):  # detections per image   for *box, conf, _, cls in det: # det: tensor.Size (bs, 7)    box: list
            labels = target_tensor[target_tensor[:, 0] == batch_idx, 1:]
            nl = len(labels)  # len of label
            tcls = labels[:, 0].tolist() if nl else []  # target class
            image_nums += 1

            # 考虑一个预测 box 都没有的情况，比如 conf 太高
            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in det]

            # # Clip boxes to image bounds TODO：有必要，因为 label 都是经过clip的，所以如果去掉clip，mAP应该会有所降低
            # clip_coords(det, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= img_tensor[batch_idx].size()[2]  # w
                tbox[:, [1, 3]] *= img_tensor[batch_idx].size()[1]  # h

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # print('stats.append: ', (correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
            '''
                        pred flag                (  [1,       0,       1,       0,       0,       1,       0,       0,       1], 
                        pred conf            tensor([0.17245, 0.14642, 0.07215, 0.07138, 0.07069, 0.06449, 0.06222, 0.05580, 0.05452]), 
                        pred cls             tensor([2.,      2.,      2.,      2.,      2.,      2.,      2.,      2.,      2.]), 
                        lb_cls                 [2.0,     2.0,  2.0, 2.0, 2.0])
            stats is a []
            '''
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))  # Append statistics (correct, conf, pcls, tcls)

    # after get stats for all images , ...
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    time.sleep(0.01)  # clw note: 防止前面 tqdm 还没输出，但是这里已经打印了
    #pf = '%20s' + '%10.3g' * 6  # print format
    pf = '%20s' + '%10s' + '%10.3g' * 5
    pf_value = pf % ('all', str(image_nums), nt.sum(), mp, mr, map, mf1)
    print(pf_value)
    if __name__ != '__main__':
        write_to_file(s, log_file_path)
        write_to_file(pf_value, log_file_path)

    results = []
    results.append( { "all" : (mp, mr, map, mf1) } )

    # Print results per class
    #if verbose and nc > 1 and len(stats):
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            #print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
            print(pf % (names[c], '', nt[c], p[i], r[i], ap[i], f1[i]))
            if __name__ != '__main__':
                write_to_file(pf % (names[c], '', nt[c], p[i], r[i], ap[i], f1[i]), log_file_path)
            results.append( { names[c] : (p[i], r[i], ap[i], f1[i]) } )

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1), maps



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/voc_yolov3.cfg', help='xxx.cfg file path')
    #parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/voc.data', help='xxx.data file path')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='0', help='device id (i.e. 0 or 0,1,2,3) ') # 默认单卡
    parser.add_argument('--src-txt-path', type=str, default='./valid.txt', help='saved img_file_paths list')
    parser.add_argument('--dst-path', type=str, default='./output', help='save detect result in this folder')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='resize to this size square and detect')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold for compute mAP')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.nms_thres,
             opt.src_txt_path,
             opt.dst_path,
             opt.weights
             )