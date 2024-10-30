import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NormalizedLinear', 'AdditiveCosineMarginLinear', 
           'AdaptiveMarginLinear', 'WeightCentralizedLinear']


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

