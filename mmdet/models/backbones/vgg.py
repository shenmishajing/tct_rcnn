from mmcv.cnn import VGG as mmcv_vgg
from ..builder import BACKBONES


@BACKBONES.register_module()
class VGG(mmcv_vgg):
    def __init__(self, pretrained = None, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.init_weights(pretrained)
