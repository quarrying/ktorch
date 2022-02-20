import torch
import torch.nn as nn

__all__ = ['LSRCrossEntropyLoss', 'OhmLsrCrossEntropyLoss', 'DiceLoss']


class LSRCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, epsilon=0.1):
        super(LSRCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        log_probs = torch.nn.functional.log_softmax(input, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        lsr_term = -log_probs.mean(dim=-1)
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * lsr_term
        return loss.mean()
    
    
class OhmLsrCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing and online hard mining.
    """
    def __init__(self, epsilon=0.1, conf_thresh=0.95):
        super(OhmLsrCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.conf_thresh = conf_thresh

    def forward(self, input, target):
        input_softmax = torch.softmax(input, -1)
        pred_conf, pred_target = torch.max(input_softmax, -1)
        mask = torch.logical_not(torch.logical_and(
            pred_target == target, pred_conf > self.conf_thresh))

        log_probs = torch.nn.functional.log_softmax(input, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        lsr_term = -log_probs.mean(dim=-1)
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * lsr_term
        if len(torch.nonzero(mask, as_tuple=False)) != 0:
            loss = loss[mask]
        return loss.mean()


class DiceLoss(nn.Module):
    """
    References:
        https://github.com/ggyyzm/pytorch_segmentation/blob/master/utils/losses.py
    """
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        """
        Notes: 假设 input 尺寸为 (N, C, H, W); target 尺寸为 (N, H, W), 而且传进来的 input 是没有归一化的.
        """
        input = torch.nn.functional.softmax(input, dim=1)
        target = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1) # one hot target
        
        input_flat = input.flatten(1, -1)
        target_flat = target.flatten(1, -1)
        intersection = (input_flat * target_flat).sum(1)
        denominator = input_flat.sum(1) + target_flat.sum(1) + self.smooth
        dice = (2. * intersection + self.smooth) / denominator
        loss = 1 - dice.mean()
        return loss

