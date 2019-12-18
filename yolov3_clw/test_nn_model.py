import torch
import cv2

from model.backbone.darknet53 import Darknet53
from model.neck.fpn import Fpn
from model.yolov3 import YOLOv3
from model.model_utils import model_info

from utils.utils import select_device

device = select_device()

if __name__ == '__main__':

    # 输入一张416x416的图片，测试输出结果
    img_np = cv2.imread('C:/Users/62349/Desktop/111/20191105_1732_0002_GC2_0000_2048.jpg')  # img_np: [nh, nw, nc]
    img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).to(device)
    # img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.float().div(255).unsqueeze(0)  # clw note： img_tensor: [batch_size, nc, nh, nw]
    # clw note：如果我们希望按批次处理图像（批量图像由 GPU 并行处理，这样可以提升速度），我们就需要固定所有图像的高度和宽度。
    #           这就需要将多个图像整合进一个大的批次（将许多 PyTorch 张量合并成一个）。

    ### 调试backbone
    net1 = Darknet53()
    print(net1)
    model_info(net1, report='full')

    ### 调试backbone+fpn
    cls_num = 1  # number of class
    net2 = Fpn([1024, 512, 256], [(cls_num + 1 + 4) * 3] * 3)
    #print(net2)
    model_info(net2, report='summary')

    ### 调试yolov3模型前向传播
    net3 = YOLOv3()
    #print(net3)
    net3 = net3.to(device)
    model_info(net3, report='summary')

    Y = net3(img_tensor)
    print('yolov3 output shape:', Y[0].shape, Y[1].shape, Y[2].shape)
    # 对于 h x w 的输入，yolov3的3个层输出shape：[batch_size, (cls_num + 1 + 4) * 3, h/8, w/8],
    #                                          [batch_size, (cls_num + 1 + 4) * 3, h/16, w/16],
    #                                          [batch_size, (cls_num + 1 + 4) * 3, h/32, w/32]
