import torch.nn as nn


def get_loss(cfg, weight=None):

   # 选择损失函数
    

    return {
        'crossentropy_loss': nn.CrossEntropyLoss(weight=weight),

    }[cfg['loss']]
