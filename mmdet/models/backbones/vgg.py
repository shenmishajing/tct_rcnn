from mmcv.cnn import VGG
from ..builder import BACKBONES

BACKBONES.register_module(module = VGG)
