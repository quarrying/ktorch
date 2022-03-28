import torch

__all__ = ['ClassifierModel']


class ClassifierModel(torch.nn.Module):
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
        
