import torch
import torchvision

__all__ = ['TvResNet18Backbone', 'TvResNet34Backbone', 'TvResNet50Backbone', 
           'TvMobileNetV2Backbone', 'TvShuffleNetV2x10Backbone', 'TvSwinTBackbone',
           'TvSwinV2TBackbone']


class TvResNet18Backbone(torch.nn.Module):
    def __init__(self, weights=None, last_stride=2, **kwargs):
        super(TvResNet18Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet18(weights=weights, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv1.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1
        self.last_channels = self.model.fc.in_features

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
        
        
class TvResNet34Backbone(torch.nn.Module):
    def __init__(self, weights=None, last_stride=2, **kwargs):
        super(TvResNet34Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet34(weights=weights, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv1.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1
        self.last_channels = self.model.fc.in_features

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
        
        
class TvResNet50Backbone(torch.nn.Module):
    def __init__(self, weights=None, last_stride=2, **kwargs):
        super(TvResNet50Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet50(weights=weights, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv2.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1
        self.last_channels = self.model.fc.in_features

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x
        
        
class TvMobileNetV2Backbone(torch.nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(TvMobileNetV2Backbone, self).__init__()
        self.model = torchvision.models.mobilenet_v2(weights=weights, **kwargs)
        self.last_channels = self.model.last_channel 

    def forward(self, x):
        x = self.model.features(x)
        return x
        
        
class TvShuffleNetV2x10Backbone(torch.nn.Module):
    def __init__(self, weights=None, **kwargs):
        super(TvShuffleNetV2x10Backbone, self).__init__()
        self.model = torchvision.models.shufflenet_v2_x1_0(weights=weights, **kwargs)
        self.last_channels = self.model._stage_out_channels[-1]

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        return x
        
        
class TvSwinTBackbone(torch.nn.Module):
    def __init__(self, weights, **kwargs):
        super(TvSwinTBackbone, self).__init__()
        self.model = torchvision.models.swin_t(weights=weights, **kwargs)
        self.last_channels = self.model.head.in_features

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        return x
        
        
class TvSwinV2TBackbone(torch.nn.Module):
    def __init__(self, weights, **kwargs):
        super(TvSwinV2TBackbone, self).__init__()
        self.model = torchvision.models.swin_v2_t(weights=weights, **kwargs)
        self.last_channels = self.model.head.in_features

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        return x
        
        
