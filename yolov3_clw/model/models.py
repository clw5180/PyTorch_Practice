from utils.parse_config import parse_model_cfg
import torch
import torch.nn as nn
import torch.nn.functional as F

class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        # 1、读取 cfg 文件，得到 module_defs（ 是一个包含多个 dict 的list，比如索引 082 对应的 dict 为
        #  <class 'dict'>: {'type': 'yolo', 'mask': '6,7,8', 'anchors': array([[ 10.,  13.],
        #        [ 16.,  30.],
        #        [ 33.,  23.],
        #        [ 30.,  61.],
        #        [ 62.,  45.],
        #        [ 59., 119.],
        #        [116.,  90.],
        #        [156., 198.],
        #        [373., 326.]]), 'classes': '80', 'num': '9', 'jitter': '.3', 'ignore_thresh': '.7', 'truth_thresh': '1', 'random': '1'}
        #    内容上和 .cfg 文件本身完全一样
        self.module_defs = parse_model_cfg(cfg)   # 注意这里除了包含索引 0-106 的所有层之外，还包含了[net]的信息，比如输入通道数channels=3，
                                                  # 这个在下一步 create_modules() 搭建conv层时会用到，所以还不能删；等进入 create_modules() 后，
                                                  # 才可以 module_defs.pop(0)；另外 get_yolo_layers() 也因此放在再下一步，也就是第三步，注意顺序不能颠倒

        # 2、将 module_defs 转化为 module_list，并且找到所有 路由层 的索引，包括 residual模块的 shortcut、
        #    有一个OrderDict类型的变量_module ，比如 key '82' 对应的 value 为 YOLOLayer()对象
        #    这里根据 1 中每个 dict， 创建相应层的对象，放入 nn.ModuleList()类型的 module_list，
        self.module_list, self.routs = create_modules(self.module_defs)

        # 3、这里存的只是三个 yolo_layer 的索引，
        #    通过遍历 module_defs 很容易得到，即 [82, 94, 106]
        self.yolo_layers = get_yolo_layers(self)
        print('yolo_layers index:', self.yolo_layers)





    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool', 'se']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]

            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        # output 是 yolo_layer的输出，注意 self.training的值会影响输出的东西
        if self.training:
            return output  # list，包含三个tensor，维度(1, 3, 13, 13, 25)，(1, 3, 26, 26, 25)，(1, 3, 52, 52, 25)，其实就是tx,ty,tx,yw
                           #  3代表一个yolo_layer含有的anchor数量
        else:
            io, p = list(zip(*output))  # output: list，包含三个tuple, 每个tuple有两个元素，0的维度是(1, 507, 25), 1的维度是(1, 3, 13, 13, 25)
                                        # io: inference output, 也就是yolo_layer的输出经过了 sigmoid 和 exp 后加上当前cell相对于feature map左上角的坐标，得到的feature map的真实坐标，然后*= self.stride得到映射到原图的真实坐标
                                        # p: training output， 也就是yolo_layer最原始的输出 tx, ty, tw, th, 也就是在当前cell内的坐标偏移
            return torch.cat(io, 1), p


def create_modules(module_defs):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        #print(i, mdef['type'])
        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(anchors=mdef['anchors'][mask],  # anchor list
                                nc=int(mdef['classes'])  # number of classes
                                )  # yolo architecture

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                # elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    #b = [-5.5, -4.0]
                b = [-5.0, -5.0]

                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
                # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
                # utils.print_model_biases(model)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'se':  # clw modify
            modules.add_module(
                'se_module',
                SELayer(output_filters[-1], reduction=int(mdef['reduction'])))
        elif mdef['type'] == 'ca':
            modules.add_module('ca', ChannelAttention(output_filters[-1],ratio=int(mdef['ratio'])))
        elif mdef['type'] == 'sa':
            modules.add_module('sa', SpatialAttention(kernel_size=int(mdef['kernelsize'])))
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs




class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc):     #  img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

    def forward(self, p, img_size, var=None):
        #bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # https://discuss.pytorch.org/t/in-pytorch-0-4-is-it-recommended-to-use-reshape-than-view-when-it-is-possible/17034
        #p = p.reshape(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2)  # prediction

        if self.training:
            return p
        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            #if 'default' in self.arc:  # seperate obj and cls
                #torch.sigmoid_(io[..., 4:])
                #torch.sigmoid_(io[..., 4])
            torch.sigmoid_(io[..., 4:])

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for classic yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ECALayer(nn.Module):

    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
