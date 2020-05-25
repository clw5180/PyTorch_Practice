import torch
import cv2

from model.backbone.darknet53 import Darknet53
from model.neck.fpn import Fpn
from yolov3 import YOLOv3
from model.model_utils import model_info

from utils.utils import select_device

device = select_device()  # 如果是单卡，可以改成最简单的一句 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    ### 输入一张416x416的图片，测试归一化之后准备送入backbone的值
    img_np = cv2.imread('C:/Users/62349/Desktop/111/20191105_1732_0002_GC2_0000_2048.jpg')  # img_np: [nh, nw, nc]
    img_np = cv2.resize(img_np, (416, 416))  # Resize to the input dimension
    img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).to(device) # TODO： img =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    # img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.float().div(255).unsqueeze(0)  # clw note： img_tensor: [batch_size, nc, nh, nw]
    # clw note：如果我们希望按批次处理图像（批量图像由 GPU 并行处理，这样可以提升速度），我们就需要固定所有图像的高度和宽度。
    #           这就需要将多个图像整合进一个大的批次（将许多 PyTorch 张量合并成一个）。

    ### 1、调试backbone
    # net1 = Darknet53()
    # print(net1)
    # model_info(net1, report='full')

    ### 2、调试backbone+fpn
    # cls_num = 1  # number of class
    # net2 = Fpn([1024, 512, 256], [(cls_num + 1 + 4) * 3] * 3)
    # #print(net2)
    # model_info(net2, report='summary')

    ### 3、测试yolov3模型前向传播
    # net3 = YOLOv3()
    # #print(net3)
    # net3 = net3.to(device)
    # model_info(net3, report='summary')
    # Y = net3(img_tensor)

    # （1）目标置信度阈值过滤
    # 对于有低于一个阈值的 objectness 分数的每个边界框，我们将其每个属性的值（表示该边界框的一整行）都设为零。
    # conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    # prediction = prediction*conf_mask

    # （2）NMS（按照类别执行）
    #我们现在拥有的边界框属性是由中心坐标以及边界框的高度和宽度决定的。但是，使用每个框的两个对角坐标能更轻松地计算两个框的
    #IoU。所以，我们可以将我们的框的(中心x, 中心 y, 高度, 宽度) 属性转换成(左上角x, 左上角y, 右下角x, 右下角y)。
    # box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    # box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    # box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    # box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    # prediction[:, :, :4] = box_corner[:, :, :4]
    # 每张图像中的「真实」检测结果的数量可能存在差异。比如，一个大小为 3 的 batch 中有 1、2、3 这 3 张图像，它们各自有 5、2、4 个「真实」检测结果。
    # 因此，一次只能完成一张图像的置信度阈值设置和 NMS。也就是说，我们不能将所涉及的操作向量化，
    # 而且必须在预测的第一个维度（包含一个 batch 中图像的索引）上循环。
    # batch_size = prediction.size(0)
    # write = False
    # for ind in range(batch_size):
    #     image_pred = prediction[ind]          #image Tensor
    #        #confidence threshholding
    #        #NMS


    ### 测试yolov3模型加载
    weight = 'C:/Users/62349/Desktop/best.pt'
    chkpt = torch.load(weight, map_location=device)
    model = YOLOv3()
    model.load_state_dict(chkpt)
    print('end!')