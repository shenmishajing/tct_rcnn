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
                 *args,
                 **kwargs):
        super(ConvFCRelationBBoxHead, self).__init__(*args, **kwargs)
        self.relation_matrix = nn.Parameter(torch.Tensor(self.fc_out_channels, self.fc_out_channels))

    def init_weights(self):
        super(ConvFCRelationBBoxHead, self).init_weights()
        nn.init.xavier_uniform_(self.relation_matrix)

    def forward(self, x, num_poses = None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            x = self.relu(self.shared_fcs[0](x))
            x = x.reshape(len(num_poses), -1, *x.shape[1:])
            x_list = []
            for i in range(len(num_poses)):
                if num_poses[i] is None or num_poses[i] <= 0:
                    x_list.append(x[i])
                    continue
                cur_x = x[i, :num_poses[i]]
                relation_weight = cur_x.mm(self.relation_matrix).mm(cur_x.T)
                relation_weight = relation_weight + torch.eye(len(relation_weight), dtype = cur_x.dtype, device = cur_x.device)
                x_list.append(torch.cat([relation_weight.mm(cur_x), x[i, num_poses[i]:]]))
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
