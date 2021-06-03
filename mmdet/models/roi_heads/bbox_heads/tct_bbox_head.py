import torch
import torch.nn as nn

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class TCTBBoxHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and relation module
     and two optional separated branches.

    .. code-block:: none

                                                    /-> cls convs -> cls fcs -> cls
        shared convs -> fc -> add delta feats -> fc
                                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self, *args, **kwargs):
        super(TCTBBoxHead, self).__init__(*args, **kwargs)
        self.fusion_module = nn.Linear(self.fc_out_channels * 2, self.fc_out_channels)

    def forward(self, x, normal_feats = None, rois = None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = self.relu(self.shared_fcs[0](x.flatten(1)))
            x = self.fusion_module(torch.cat([x, normal_feats], dim = 1))

            for fc in self.shared_fcs[1:]:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
