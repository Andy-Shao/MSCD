import torch
from torch import nn

def mse_loss(o1:torch.Tensor, o2:torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    o1 = F.softmax(o1, dim=1)
    o2 = F.softmax(o2, dim=1)
    l2_norm = torch.sqrt(torch.sum(torch.pow(o1 - o2, 2.0), dim=1))
    mse_loss = torch.mean(l2_norm, dim=0)
    return mse_loss

def SoftCrossEntropyLoss(logit: torch.Tensor, soft_pseudo_label: torch.Tensor) -> torch.Tensor:   # Checked and is correct
    """Pseudo-label cross-entropy loss uses this loss function"""
    import torch.nn.functional as F
    percentage = F.log_softmax(logit, dim=1)
    # print(f'left shape: {soft_pseudo_label.shape}, right shape: {percentage.shape}')
    return -(soft_pseudo_label * percentage).sum(dim=1)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, reduction=True, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon 
        self.reduction = reduction 
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.use_gpu = use_gpu
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1) # cross-entropy loss
        if self.reduction:
            return loss.mean()
        else:
            return loss