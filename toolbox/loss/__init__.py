import torch.nn as nn
from toolbox.loss.loss import DiceLoss


def get_loss(cfg, weight=None):

   # 选择损失函数
    

    return {
        'crossentropy_loss': nn.CrossEntropyLoss(weight=weight),

    }[cfg['loss']]
