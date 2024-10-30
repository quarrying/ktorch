import torch
import torch.nn as nn

from .pooling import GlobalAttentionPooling

__all__ = ['ClassifierModel', 'GpBnFcBn', 'GpLnFcLn', 'CCLNorm2d']


class ClassifierModel(nn.Module):
    def __init__(self, backbone, head):
        super(ClassifierModel, self).__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x, label=None):
        x = self.backbone(x)
        if label is None:
            x = self.head(x)
        else:
            x = self.head(x, label)
        return x
        

class GpBnFcBn(nn.Module):
    def __init__(self, in_channels, embedding_size=512, pooling_type='gap'):
        super(GpBnFcBn, self).__init__()
        if pooling_type.lower() == 'gmp':
            self.gp = nn.AdaptiveMaxPool2d(1)
        elif pooling_type.lower() == 'gap':
            self.gp = nn.AdaptiveAvgPool2d(1)
        elif pooling_type.lower() in ['global_attention_pooling', 'gp_attn']:
            self.gp = GlobalAttentionPooling()
        else:
            raise ValueError('Unsupported pooling_type!')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn1.bias.requires_grad_(False)
        self.fc = nn.Conv2d(in_channels, embedding_size, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(embedding_size)
        self.bn2.bias.requires_grad_(False)
 
    def forward(self, x):
        x = self.gp(x)
        x = self.bn1(x)
        x = self.fc(x)
        x = self.bn2(x)
        return x


class GpLnFcLn(nn.Module):
    def __init__(self, in_channels, embedding_size=512, pooling_type='gap'):
        super(GpLnFcLn, self).__init__()
        if pooling_type.lower() == 'gmp':
            self.gp = nn.AdaptiveMaxPool2d(1)
        elif pooling_type.lower() == 'gap':
            self.gp = nn.AdaptiveAvgPool2d(1)
        elif pooling_type.lower() == 'global_attention_pooling':
            self.gp = GlobalAttentionPooling()
        else:
            raise ValueError('Unsupported pooling_type!')
        self.ln1 = nn.LayerNorm((in_channels, 1, 1))
        self.ln1.bias.requires_grad_(False)
        self.fc = nn.Conv2d(in_channels, embedding_size, 1, bias=False)
        self.ln2 = nn.LayerNorm((in_channels, 1, 1))
        self.ln2.bias.requires_grad_(False)
        
    def forward(self, x):
        x = self.gp(x)
        x = self.ln1(x)
        # 这里不用残差连接, 是因为 fc 层的输入输出通道数不匹配
        # 因为没有用残差连接, 所以没有 PreNorm 和 PostNorm 之分
        x = self.fc(x)
        x = self.ln2(x)
        return x
    
    
class CCLNorm2d(torch.nn.BatchNorm2d):
    """
    References:
        [2018] Face Recognition via Centralized Coordinate Learning
    """
    def __init__(self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__(num_features, eps, momentum, False, track_running_stats, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight_scalar = torch.nn.Parameter(torch.empty(1, **factory_kwargs))
        
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        torch.nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        # passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        # used for normalization (i.e. in eval mode when buffers are not None).
        weight_vector = torch.full_like((self.num_features,), self.weight_scalar.item())
        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight_vector,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

