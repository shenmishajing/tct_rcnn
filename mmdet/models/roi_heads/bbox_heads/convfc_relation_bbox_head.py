import torch
import torch.nn as nn

from mmdet.models.builder import HEADS
from .convfc_bbox_head import Shared2FCBBoxHead


@HEADS.register_module()
class ConvFCRelationBBoxHead(Shared2FCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and relation module
     and two optional separated branches.

    .. code-block:: none

                                                    /-> cls convs -> cls fcs -> cls
        shared convs -> fc -> relation module -> fc
                                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_relation_parts = 1,
                 *args,
                 **kwargs):
        super(ConvFCRelationBBoxHead, self).__init__(*args, **kwargs)
        self.num_relation_parts = num_relation_parts
        assert self.fc_out_channels % self.num_relation_parts == 0, 'fc_out_channels must be divisible by num_relation_parts'
        part_len = self.fc_out_channels // self.num_relation_parts
        self.relation_matrix = nn.Parameter(torch.Tensor(self.num_relation_parts, part_len, part_len))
        self.softmax = nn.Softmax(dim = -1)

    def init_weights(self):
        super(ConvFCRelationBBoxHead, self).init_weights()
        nn.init.xavier_uniform_(self.relation_matrix)

    def _relation_forwards(self, x):
        x = x.reshape(x.shape[0], self.num_relation_parts, -1)
        x = x.permute(1, 0, 2)
        encoding_feat = x.bmm(self.relation_matrix)
        relation_weight = encoding_feat.bmm(encoding_feat.permute(0, 2, 1))
        relation_weight = self.softmax(relation_weight)
        relation_feature = (relation_weight.bmm(x) + x) / 2
        relation_feature = relation_feature.permute(1, 0, 2)
        return relation_feature.reshape(relation_feature.shape[0], -1)

    def forward(self, x, rois = None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            x = self.relu(self.shared_fcs[0](x))

            if rois is None:
                x = self._relation_forwards(x)
            else:
                x_list = []
                for i in rois[:, 0].unique():
                    x_list.append(self._relation_forwards(x[rois[:, 0] == i]))
                x = torch.cat(x_list)

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
