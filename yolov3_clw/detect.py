from utils.utils import select_device
from model.models import Darknet
from utils.datasets import VocDataset
from utils.utils import non_max_suppression, plot_one_box, load_classes
from utils.parse_config import parse_data_cfg

import argparse
import torch
from torch.utils.data import DataLoader
import cv2
import os
from tqdm import tqdm
import time

def detect():
    # 0、初始化一些参数
    cfg = opt.cfg
    weights = opt.weights
    src_txt_path = opt.src_txt_path
    img_size = opt.img_size
    batch_size = opt.batch_size
    dst_path = opt.dst_path
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    device = select_device(opt.device)
    classes = load_classes(parse_data_cfg(opt.data)['names'])

    # 1、加载网络
    model = Darknet(cfg)
    if weights.endswith('.pt'):      # TODO: .weights权重格式
        model.load_state_dict(torch.load(weights)['model'])  # TODO：map_location=device ？
    model.to(device).eval()

    # 2、加载数据集
    test_dataset = VocDataset(src_txt_path, img_size, is_training=False)
    dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,    # TODO
                            collate_fn=test_dataset.test_collate_fn)   # TODO

    # 3、预测，前向传播
    start = time.time()
    pbar = tqdm(dataloader)
    for i, (img_tensor, img0, img_name) in enumerate(pbar):
        pbar.set_description("Already Processed %d image: " % (i+1))
        # print('clw: Already Processed %d image' % (i+1))
        img_tensor = img_tensor.to(device)   # (bs, 3, 416, 416)
        pred = model(img_tensor)[0]   # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        for batch_idx, det in enumerate(pred):  # detections per image
            if det is not None:   # and len(det):  # clw note: important !
                #or box in det:
                for *box, conf, _, cls in det: # det: tensor.Size (bs, 7)    box: list
                    orig_h, orig_w = img0[batch_idx].shape[:2]  # 坐标变换
                    new_h = new_w = img_tensor.size()[2]  # 绘图，resize后的图的框 -> 原图的框，new -> orig
                    ratio_h = orig_h / new_h
                    ratio_w = orig_w / new_w
                    x1 = int(ratio_w * box[0])
                    y1 = int(ratio_h * box[1])
                    x2 = int(ratio_w * (box[2]))
                    y2 = int(ratio_h * (box[3]))
                    label = '%s %.2f' % (classes[int(cls)], conf)

                    # 预测结果可视化
                    plot_one_box([x1, y1, x2, y2], img0[batch_idx], label=label, color=(255, 0, 0))
                    #cv2.rectangle(img0[batch_idx], (x1, y1), (x2, y2), (0, 0, 255), 1)  # 如果报错 TypeError: an integer is required (got type tuple)，检查是不是传入了img_tensor


            # 保存结果
            cv2.imwrite(os.path.join(dst_path, img_name[batch_idx]), img0[batch_idx])

    print('time use: %.3fs' %(time.time() - start))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='xxx.cfg file path')
    parser.add_argument('--data', type=str, default='cfg/coco.data', help='xxx.data file path')
    parser.add_argument('--device', type=str, default='0', help='device id (i.e. 0 or 0,1,2,3) ') # 默认单卡
    parser.add_argument('--src-txt-path', type=str, default='./valid.txt', help='saved img_file_paths list')
    parser.add_argument('--dst-path', type=str, default='./output', help='save detect result in this folder')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='resize to this size square and detect')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression, do not be confused with the iou threshold for mAP')
    parser.add_argument('--batch-size', type=int, default=4)
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()