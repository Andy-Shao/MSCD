import torch
from torch import nn, optim

def build_optimizer(lr:float, auT:nn.Module, auC:nn.Module=None, auT_decay:float=1.0, auC_decay:float=1.0) -> optim.Optimizer:
    param_group = []
    learning_rate = lr
    for k, v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * auT_decay}]
    if auC is not None:
        for k, v in auC.named_parameters():
            param_group += [{'params':v, 'lr':learning_rate * auC_decay}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(
        optimizer: torch.optim.Optimizer, epoch:int, lr_cardinality:int, gamma=10, power=0.75, threshold=1,
        momentum:float=.9
    ) -> optim.Optimizer:
    if epoch >= lr_cardinality-threshold:
        return optimizer
    decay = (1 + gamma * epoch / lr_cardinality) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = momentum
        param_group['nestenv'] = True
    return optimizer