import torch
import torchvision

__all__ = ['TvResNet18Backbone', 'TvResNet34Backbone', 'TvResNet50Backbone', 
           'TvMobileNetV2Backbone', 'TvShuffleNetV2x10backbone']


class TvResNet18Backbone(torch.nn.Module):
    def __init__(self, pretrained=True, last_stride=2, **kwargs):
        super(TvResNet18Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet18(pretrained=pretrained, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv1.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1

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
    def __init__(self, pretrained=True, last_stride=2, **kwargs):
        super(TvResNet34Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet34(pretrained=pretrained, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv1.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1

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
    def __init__(self, pretrained=True, last_stride=2, **kwargs):
        super(TvResNet50Backbone, self).__init__()
        assert last_stride in [1, 2]
        self.model = torchvision.models.resnet50(pretrained=pretrained, **kwargs)
        if last_stride == 1:
            self.model.layer4[0].conv2.stride = 1
            self.model.layer4[0].downsample[0].stride = 1
            self.model.layer4[0].stride = 1
        
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
    def __init__(self, pretrained, **kwargs):
        super(TvMobileNetV2Backbone, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained, **kwargs)

    def forward(self, x):
        x = self.model.features(x)
        return x
        
        
class TvShuffleNetV2x10backbone(torch.nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(TvShuffleNetV2x10backbone, self).__init__()
        self.model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained, **kwargs)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        return x
        
        