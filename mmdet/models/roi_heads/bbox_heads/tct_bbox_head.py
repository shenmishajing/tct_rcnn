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

    def __init__(self,
                 *args,
                 **kwargs):
        super(TCTBBoxHead, self).__init__(*args, **kwargs)
        self.dynamic_fc = nn.Linear(self.in_channels, 2 * self.in_channels)

    def forward(self, x, normal_feats = None, batch_inds = None):
        x_list = []
        for i in range(len(normal_feats)):
            normal_feat = normal_feats[i]
            if normal_feat.ndim == 4:
                normal_feat = normal_feat.mean(0)
            normal_feat = normal_feat.flatten(1).T
            params = self.dynamic_fc(normal_feat).reshape(self.roi_feat_area, self.in_channels, 2)
            cur_x = x[batch_inds == i].flatten(2).permute(0, 2, 1)
            cur_x = cur_x.matmul(params[:, :, 0].T).matmul(params[:, :, 1]).permute(0, 2, 1).reshape(-1, self.in_channels,
                                                                                                     *self.roi_feat_size)
            x_list.append(cur_x)
        x = torch.cat(x_list)
        return super(TCTBBoxHead, self).forward(x)
