import torch
import torch.nn as nn

__all__ = ['ClassifierModel', 'GpBnFcBn']


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

