from torch import nn
from torch.nn import Upsample
from ..model_utils import Convolutional, Route


class Fpn(nn.Module):
    def __init__(self, in_channels, out_channels):  # 默认 [1024, 512, 256] 和 [(cls_num + 1 + 4)*3] *3
        super().__init__()

        in_channel1, in_channel2, in_channel3 = in_channels
        out_channel1, out_channel2, out_channel3 = out_channels

        # yolo_layer1
        self.conv1_0 = nn.Sequential(
            Convolutional(in_channel1, 512, 1, 0),  # Y_74
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, 512, 1, 0),
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, 512, 1, 0),
        )  # 输出Y_79，即input1 of yolo_layer1
           # 之后上采样然后concat，在yolo_layer2中也要被使用，因此需要单独提出来
        self.conv1_1 = nn.Sequential(
            Convolutional(512, 1024, 3, 1),
            Convolutional(1024, out_channel1, 1, 0),  # clw note: 这里out_channels = (cls_num + 4 + 1) * 3, cls_num就是类别数
                                                      #           在 YOLO 中，预测是通过卷积层完成的，它是一个全卷积神经网络，
                                                      #           其核心尺寸为：1×1×（B×（5+C）），其中C就是上面的cls_num，
                                                      #           B是每层的anchor个数，这里是9 / 3 = 3
            # TODO：上面貌似没有bn和activate, 所以应该是nn.conv2D??
        )
        # 输出Y_81，即output1 of yolo_layer1

        # yolo_layer2
        self.conv2_0 = nn.Sequential(
            Convolutional(in_channel2, 256, 1, 0),
            Upsample(scale_factor=2, mode='nearest'),
        )
        self.route2_1 = Route()
        self.conv2_2 = nn.Sequential(
            Convolutional(768, 256, 1, 0),
            Convolutional(256, 512, 3, 1),
            Convolutional(512, 256, 1, 0),
            Convolutional(256, 512, 3, 1),
            Convolutional(512, 256, 1, 0),
        )
        self.conv2_3 = nn.Sequential(
            Convolutional(256, 512, 3, 1),
            Convolutional(512, out_channel2, 1, 0),
        )

        # yolo_layer3
        self.conv3_0 = nn.Sequential(
            Convolutional(in_channel3, 128, 1, 0),
            Upsample(scale_factor=2, mode='nearest'),
        )
        self.route3_1 = Route()  # Y_98
        self.conv3_2 = nn.Sequential(
            Convolutional(384, 128, 1, 0),
            Convolutional(128, 256, 3, 1),
            Convolutional(256, 128, 1, 0),
            Convolutional(128, 256, 3, 1),
            Convolutional(256, 128, 1, 0),
            Convolutional(128, 256, 3, 1),
            Convolutional(256, out_channel3, 1, 0),
        )

    def forward(self, Y_36, Y_61, Y_74):
        Y_79 = self.conv1_0(Y_74)  # input1 of yolo_layer1
        Y_81 = self.conv1_1(Y_79)  # output1 of yolo_layer1
        Y_85 = self.conv2_0(Y_79)
        Y_86 = self.route2_1(Y_61, Y_85)
        Y_91 = self.conv2_2(Y_86)  # input2 of yolo_layer2
        Y_93 = self.conv2_3(Y_91)  # output2 of yolo_layer2
        Y_97 = self.conv3_0(Y_91)
        Y_98 = self.route3_1(Y_36, Y_97)
        Y_105 = self.conv3_2(Y_98)  # out3, from yolo_layer3
        return Y_105, Y_93, Y_81