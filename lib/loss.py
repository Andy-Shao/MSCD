from typing import Literal

import torch
from torch import nn

class ContrastiveLoss(nn.Module):
    def __init__(
        self, hi_def_smth:float, class_num:int, device:str, eps:float=1e-8, 
        dist:Literal['cos_sim', 'l2', 'sq_l2']='l2', tau:float=1.
    ):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.hi_def_smth = hi_def_smth
        self.class_num = class_num
        self.device = device
        self.eps = eps
        self.dist = dist
        self.tau = tau

    def __marking__(self, outs:torch.Tensor, pseudo_labels:torch.Tensor) -> torch.Tensor:
        hi_mark = (1-self.hi_def_smth)*torch.eye(self.class_num)[0] + self.hi_def_smth/self.class_num
        hi_mark_val, _ = torch.max(hi_mark, dim=0)
        pl_val, _ = torch.max(pseudo_labels.detach(), dim=1)
        is_hi_mark = (pl_val >= hi_mark_val)
        hi_marks = (is_hi_mark.clone().unsqueeze(dim=1) & is_hi_mark.unsqueeze(dim=0)).float()

        _, out_pos = torch.max(outs.detach(), dim=1)
        same_pred_marks = (out_pos.unsqueeze(dim=1) == out_pos.unsqueeze(dim=0)).float() * 2 - 1

        marks = hi_marks * same_pred_marks
        marks[(marks==-0.) & torch.signbit(marks)] = 0. # covert -0.0 to 0.0
        marks = marks.fill_diagonal_(fill_value=0.) # fill leading-diagonal to 0.
        return marks

    def forward(self, x:torch.Tensor, pseudo_labels:torch.Tensor) -> torch.Tensor:
        if self.dist == 'cos_sim': # consine similarity
            x_norm = nn.functional.normalize(x, p=2, dim=1)
            dist_val = x_norm @ x_norm.T # self-consine similarity
        elif self.dist == 'l2': # L2-norm
            x_norm = nn.functional.normalize(x, p=2, dim=1)
            dist_val = torch.cdist(x1=x_norm, x2=x_norm, p=2)
        elif self.dist == 'sq_l2': # squared L2-norm
            x_norm = nn.functional.normalize(x, p=2, dim=1)
            dist_val = 2 - 2 * (x_norm @ x_norm.T)
            dist_val = dist_val.clamp_min_(0)
        if self.tau != 1.: dist_val = dist_val / self.tau
        marks = self.__marking__(outs=x, pseudo_labels=pseudo_labels)
        mark_norm = marks / (marks.abs().sum(dim=1, keepdim=True)+self.eps)
        loss = mark_norm * self.log_softmax(dist_val)
        loss = loss.sum(dim=1)
        loss = loss.mean()
        if self.dist == 'cos_sim': loss = - loss
        return loss

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
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1) # cross-entropy loss
        if self.reduction:
            return loss.mean()
        else:
            return loss