import torch
import numpy as np

__all__ = ['DepthSeparableConv2d', 'CoordConv2d']


class DepthSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=True):
        super(DepthSeparableConv2d, self).__init__()

        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, dilation=dilation, 
                                         groups=in_channels, bias=bias)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                         stride=1, padding=0, dilation=1, groups=1, bias=bias)
    
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CoordConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CoordConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.uv = None

    def forward(self, x):
        if self.uv is None:
            height, width = x.shape[2], x.shape[3]
            u, v = np.meshgrid(range(width), range(height))
            u = 2 * u / (width - 1) - 1
            v = 2 * v / (height - 1) - 1
            uv = np.stack((u, v)).reshape(1, 2, height, width)
            self.uv = torch.from_numpy(uv.astype(np.float32))
        self.uv = self.uv.to(x.device)
        uv = self.uv.expand(x.shape[0], *self.uv.shape[1:])
        xuv = torch.cat((x, uv), dim=1)
        y = self.conv(xuv)
        return y

