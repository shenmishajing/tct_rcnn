import torch.nn as nn

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class ComparisonBBoxHead(ConvFCBBoxHead):
    """Shared module bbox head for Comparison Detector.
    """

    def __init__(self, in_channels = 256, reg_class_agnostic = False, *args, **kwargs):
        super(ComparisonBBoxHead, self).__init__(
            in_channels = 256,
            num_shared_convs = 1,
            num_shared_fcs = 0,
            num_cls_convs = 0,
            num_cls_fcs = 0,
            num_reg_convs = 0,
            num_reg_fcs = 0,
            reg_class_agnostic = False,
            *args,
            **kwargs)
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, 1)
        if self.with_reg:
            self.fc_reg = nn.Linear(self.reg_last_dim, 4)
        self.conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, reference):
        # shared part
        N = x.shape[0]
        x = (x[:, None, ...] - reference[None, ...]) ** 2
        x = x.flatten(0, 1)
        x = self.conv(x)
        cls_score, bbox_pred = super(ComparisonBBoxHead, self).forward(x)
        return -cls_score.reshape(N, -1), bbox_pred.reshape(N, -1)
