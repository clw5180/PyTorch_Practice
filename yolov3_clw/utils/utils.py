import os
import torch

def select_device(device):  # 暂时不支持 CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
    assert torch.cuda.is_available(), 'CUDA unavailable and CPU not support yet, invalid device: %s' % device  # check availablity

    nums_of_gpu = torch.cuda.device_count()
    #if nums_of_gpu > 1 and batch_size:    # TODO: 多卡，batch_size不能被卡的总数整除 check that batch_size is compatible with device_count
    #    assert batch_size % nums_of_gpu == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, nums_of_gpu)
    x = [torch.cuda.get_device_properties(i) for i in range(nums_of_gpu)]
    s = 'Using CUDA'
    for i in range(0, nums_of_gpu):
        if i == 1:
            s = ' ' * len(s)
        print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" % (s, i, x[i].name, x[i].total_memory / 1024 ** 2))  # bytes to MB


    '''
    Using CUDA device0 _CudaDeviceProperties(name='GeForce GTX 1080', total_memory=8116MB)
           device1 _CudaDeviceProperties(name='GeForce GTX 1080', total_memory=8119MB)
    '''

    return torch.device('cuda:0')


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
