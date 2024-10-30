import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .others import BatchL2Norm


__all__ = ['NormalizedLinear', 'AdditiveCosineMarginLinear', 
           'AdaptiveMarginLinear', 'WeightCentralizedLinear',
           'WeightL2NormalizedLinear', 'AdaptiveNormalizedLinear']


class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(NormalizedLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input): 
        return F.linear(F.normalize(input, dim=1), F.normalize(self.weight, dim=1))
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
        
        
class AdditiveCosineMarginLinear(nn.Module):
    """Additive Cosine Margin Linear

    References:
        [2018] Additive margin softmax for face verification
        [2018] CosFace_ Large Margin Cosine Loss for Deep Face Recognition
    """
    def __init__(self, in_features, out_features, margin, scale=30, device=None, dtype=None):
        super(AdditiveCosineMarginLinear, self).__init__()
        self.scale = scale
        self.margin = margin
        self.normalized_linear = NormalizedLinear(in_features, out_features, 
                                                  device=device, dtype=dtype)
        
    def forward(self, input, label):
        output_cos = self.normalized_linear(input)
        if self.training:
            # # Method I, to be tested
            # margin_matrix = torch.zeros_like(output_cos)
            # margin_matrix.scatter_(1, label.view(-1, 1), self.margin)
            # output = self.scale * (output_cos - margin_matrix)
            
            # # Method II, to be tested
            # output_cos.scatter_(1, label.view(-1, 1), -self.margin, reduce='add')
            # output = output_cos * self.scale
            
            one_hot_labels = torch.zeros_like(output_cos)
            one_hot_labels.scatter_(1, label.view(-1, 1), 1.0)
            output = self.scale * (output_cos - one_hot_labels * self.margin)
        else:
            output = self.scale * output_cos
        return output
        
        
class AdaptiveMarginLinear(nn.Module):
    def __init__(self, in_features, out_features, gamma, scale=30, device=None, dtype=None):
        super(AdaptiveMarginLinear, self).__init__()
        assert 0 < gamma <= 1, 'gamma must be in (0, 1]'
        self.scale = scale
        self.gamma = gamma
        self.normalized_linear = NormalizedLinear(in_features, out_features, 
                                                  device=device, dtype=dtype)
        
    def forward(self, input, label):
        output_cos = self.normalized_linear(input)
        if self.training:
            with torch.no_grad():
                margin = (1 - self.gamma) * (1 - output_cos)
                margin.scatter_(1, label.view(-1, 1), 0.0)
            output = self.scale * (output_cos + margin)
        else:
            output = self.scale * output_cos
        return output
        

class WeightCentralizedLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(WeightCentralizedLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), 
                                   **factory_kwargs))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input): 
        mean = torch.mean(self.weight, dim=0, keepdim=True)
        return F.linear(input, self.weight - mean)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class WeightL2NormalizedLinear(nn.Module):
    """
    Notes:
        用在分类层, 使所有类的权重一样
    """
    def __init__(self, in_features, out_features, eps: float = 1e-12, 
                 do_centralize=False, device=None, dtype=None):
        super(WeightL2NormalizedLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.do_centralize = do_centralize
        self.weight = nn.Parameter(torch.empty((out_features, in_features), 
                                   **factory_kwargs))
        self.reset_parameters()
        
    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input): 
        if self.do_centralize:
            # Centralized 是在第 0 维上做的, L2 Normalized 是在第 1 维上做的
            mean = torch.mean(self.weight, dim=0, keepdim=True)
            centralized_weight = self.weight - mean
        else:
            centralized_weight = self.weight
            
        norm = centralized_weight.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps).expand_as(input)
        mean_norm = torch.mean(norm)
        weight = mean_norm * centralized_weight / norm
        return F.linear(input, weight)
    
    
class AdaptiveNormalizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, momentum: float = 0.1, eps=1e-12, device=None, dtype=None):
        super(AdaptiveNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_l2_norm = BatchL2Norm(momentum=momentum, eps=eps, device=device, dtype=dtype)
        self.weight_l2_norm = WeightL2NormalizedLinear(in_features, out_features, eps=eps, device=device, dtype=dtype)

    def forward(self, input):
        return self.weight_l2_norm(self.batch_l2_norm(input))
    
    