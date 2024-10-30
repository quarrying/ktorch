import torch
import torch.nn as nn

from .pooling import GlobalAttentionPooling

__all__ = ['ClassifierModel', 'GpBnFcBn', 'GpLnFcLn']


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
    
    