from toolbox.models.unet import unet
from toolbox.models.segnet import SegNet
from toolbox.models.linknet import linknet
from toolbox.models.enet import ENet
from toolbox.models.deeplabv3plus import DeepLabv3_plus
from toolbox.models.bisenet import BiSeNet


def get_model(cfg):
    return {
        'unet': unet,
        'segnet': SegNet,
        'linknet': linknet,
        'enet': ENet,
        'bisenet': BiSeNet,
        'deeplabv3plus': DeepLabv3_plus,

    }[cfg['model_name']](n_classes=cfg['n_classes'])
