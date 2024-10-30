import torch

__all__ = ['GMAP', 'MixedPool2d', 'AdaptiveMixedPool2d', 
           'global_attention_pooling', 'GlobalAttentionPooling']


class GMAP(torch.nn.Module):
    def __init__(self):
        super(GMAP, self).__init__()
        self.gmp = torch.nn.AdaptiveMaxPool2d(1)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        
    def forward(self, inputs):
        gmp_outputs = self.gmp(inputs)
        gap_outputs = self.gap(inputs)
        return gmp_outputs + gap_outputs


class MixedPool2d(torch.nn.Module):
    """
    References:
        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, 
                 input_channels=None, device=None, dtype=None):
        super(MixedPool2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)
        if input_channels is None:
            w_shape = (1,)
        else:
            w_shape = (1, input_channels, 1, 1)
        self.weight = torch.nn.Parameter(torch.zeros(w_shape, **factory_kwargs))
        
    def forward(self, x):
        y_max = self.max_pool2d(x)
        y_avg = self.avg_pool2d(x)
        weight = torch.sigmoid(self.weight)
        return y_max + weight * (y_avg - y_max)
        
    
class AdaptiveMixedPool2d(torch.nn.Module):
    """
    References:
        torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
        torch.nn.AdaptiveMaxPool2d(output_size)
    """
    def __init__(self, output_size, input_channels=None, device=None, dtype=None):
        super(AdaptiveMixedPool2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.gmp = torch.nn.AdaptiveMaxPool2d(output_size)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size)
        if input_channels is None:
            w_shape = (1,)
        else:
            w_shape = (1, input_channels, 1, 1)
        self.weight = torch.nn.Parameter(torch.zeros(w_shape, **factory_kwargs))
        
    def forward(self, x):
        y_max = self.gmp(x)
        y_avg = self.gap(x)
        weight = torch.sigmoid(self.weight)
        return y_max + weight * (y_avg - y_max)
        
        
def global_attention_pooling(x, scale=1, keepdim=True):
    assert x.ndim >= 3
    # (..., C, H * W)
    x = x.flatten(-2)
    # (..., C, H * W)
    attn = torch.softmax(x * scale, dim=-1)
    # (..., C)
    attn_gp = torch.sum(x * attn, dim=-1)

    if keepdim:
        # (..., C, 1, 1)
        attn_gp = attn_gp.unsqueeze(-1).unsqueeze(-1)
    return attn_gp


class GlobalAttentionPooling(torch.nn.Module):
    def __init__(self, scale=1):
        super(GlobalAttentionPooling, self).__init__()
        self.scale = scale
        
    def forward(self, x):
        return global_attention_pooling(x, self.scale, keepdim=True)
    
    