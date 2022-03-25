import torch

__all__ = ['SharedBatchNorm1d', 'ClassifierModel']


class SharedBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
        with torch.no_grad():
            gamma_mean = torch.mean(self.weight)
            self.weight.fill_(gamma_mean)
        output = super(SharedBatchNorm1d, self).forward(input)
        return output


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
        
