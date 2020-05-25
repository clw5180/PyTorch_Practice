import torch
from torch import nn
from model.backbone.darknet53 import Darknet53
from model.neck.fpn import Fpn
from model.head.yolo_head import Yolo_head
import cfg.cfg as cfg

class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_num = 1
        self.out_channels = (self.cls_num + 1 + 4) * cfg.MODEL["ANCHORS_PER_SCLAE"]
        self.backbone = Darknet53()
        self.fpn = Fpn(in_channels=[1024, 512, 256], out_channels=[self.out_channels] * 3)

        self.anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(cfg.MODEL["STRIDES"])

        self.yolo_head1 = Yolo_head(cls_num=self.cls_num, anchors=self.anchors[0], stride=self.strides[0])
        self.yolo_head2 = Yolo_head(cls_num=self.cls_num, anchors=self.anchors[1], stride=self.strides[0])
        self.yolo_head3 = Yolo_head(cls_num=self.cls_num, anchors=self.anchors[2], stride=self.strides[0])

    def forward(self, X):
        Y_36, Y_61, Y_74 = self.backbone(X)
        Y_81, Y_93, Y_105 = self.fpn(Y_74, Y_61, Y_36)
        yolo_head1_out = self.yolo_head1(Y_81)  # smallest feature map, detect large object
        yolo_head2_out = self.yolo_head2(Y_93)
        yolo_head3_out = self.yolo_head3(Y_105)
        # print('yolov3 output shape:', yolo_head1_out.shape, yolo_head2_out.shape, yolo_head3_out.shape)
        # 对于 h x w 的输入，yolov3的3个层输出shape：[batch_size, (cls_num + 1 + 4) * 3, h/32, w/32],
        #                                           [batch_size, (cls_num + 1 + 4) * 3, h/16, w/16],
        #                                           [batch_size, (cls_num + 1 + 4) * 3, h/8, w/8]
        results = torch.cat((yolo_head1_out, yolo_head2_out, yolo_head3_out), dim=1)   # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
                                                                                       # 其中507=13*13*3， 2028=26*26*3
        return results    # results: [1, 10647, 85]

