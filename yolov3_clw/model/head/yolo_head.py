# YOLO的输出是一个卷积特征图，它包含沿特征图深度的边界框属性。
# 单元格预测的边界框属性被相互堆叠在一起。因此，如果要访问（5,6）处单元格的第二个边界框，
# 则需要通过map [5,6，（5 + C）：2 *（5 + C）]对它进行索引。
# 这种形式对输出处理（例如根据目标置信度进行阈值处理，向中心坐标添加网格偏移量，应用锚等）非常不方便。

from torch import nn
from ..model_utils import predict_transform

class Yolo_head(nn.Module):
    def __init__(self, cls_num, anchors, stride):  # 3个yolo层的stride分别为8, 16, 32
        super().__init__()

        self.cls_num = cls_num
        self.anchors = anchors
        self.stride = stride

    def forward(self, prediction):
        yolo_head_out = predict_transform(prediction, num_classes=self.cls_num, anchors=self.anchors, stride=self.stride)     
        return yolo_head_out





