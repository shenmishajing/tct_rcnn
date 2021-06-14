import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead


@HEADS.register_module()
class TCTBBoxHead(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and relation module
     and two optional separated branches.

    .. code-block:: none

                               /-> cls convs -> cls fcs -> cls
        shared convs -> 2 * fc 
                               \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 pos_bboxes_temperature = 10,
                 loss_compare = dict(loss_weight = 0.1),
                 *args,
                 **kwargs):
        self.pos_bboxes_temperature = pos_bboxes_temperature
        self.loss_compare = loss_compare
        super(TCTBBoxHead, self).__init__(*args, **kwargs)
        self.dynamic_fc = nn.Linear(self.in_channels, 2 * self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.norm1 = nn.LayerNorm([self.roi_feat_area, self.roi_feat_area])
        self.norm2 = nn.LayerNorm([self.roi_feat_area, self.in_channels])
        self.norm3 = nn.LayerNorm([self.in_channels, *self.roi_feat_size])

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat = True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg = rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        pos_gt_labels = torch.cat(pos_gt_labels_list)
        pos_bbox_label_matrix = pos_gt_labels[:, None] == pos_gt_labels[None, :]
        pos_bbox_label_matrix.fill_diagonal_(False)
        return labels, label_weights, bbox_targets, bbox_weights, pos_bbox_label_matrix

    @force_fp32(apply_to = ('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             pos_bbox_feats,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             pos_bbox_label_matrix,
             reduction_override = None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor = avg_factor,
                    reduction_override = reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor = bbox_targets.size(0),
                    reduction_override = reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        if pos_bbox_feats is not None:
            pos_bbox_feats = pos_bbox_feats / torch.norm(pos_bbox_feats, dim = 0, keepdim = True)
            pos_bbox_relation_matrix = pos_bbox_feats.mm(pos_bbox_feats.T) / self.pos_bboxes_temperature
            pos_bbox_relation_matrix = pos_bbox_relation_matrix - torch.diag_embed(torch.diag(pos_bbox_relation_matrix))
            pos_bbox_relation_matrix_exp = torch.exp(pos_bbox_relation_matrix)
            pos_bbox_relation_matrix_exp = pos_bbox_relation_matrix_exp - torch.diag_embed(torch.diag(pos_bbox_relation_matrix_exp))
            pos_bbox_loss_matrix = pos_bbox_relation_matrix_exp / torch.sum(pos_bbox_relation_matrix_exp, dim = 0, keepdim = True)
            pos_bbox_loss_matrix = pos_bbox_loss_matrix - torch.diag_embed(torch.diag(pos_bbox_loss_matrix) - 1)
            pos_bbox_loss_matrix = -torch.log(pos_bbox_loss_matrix) * pos_bbox_label_matrix
            losses['loss_compare'] = self.loss_compare['loss_weight'] * torch.sum(pos_bbox_loss_matrix) / torch.sum(
                pos_bbox_loss_matrix != 0)
        return losses

    def forward(self, x, normal_feats = None, batch_inds = None):
        x_list = []
        for i in range(len(normal_feats)):
            normal_feat = normal_feats[i]
            if normal_feat.ndim == 4:
                normal_feat = normal_feat.mean(0)
            normal_feat = normal_feat.flatten(1).T
            params = self.dynamic_fc(normal_feat).reshape(self.roi_feat_area, self.in_channels, -1)
            cur_x = x[batch_inds == i].flatten(2).permute(0, 2, 1)
            cur_x = self.relu(self.norm1(cur_x.matmul(params[:, :, 0].T)))
            cur_x = self.relu(self.norm2(cur_x.matmul(params[:, :, 1])))
            cur_x = x[batch_inds == i] + cur_x.permute(0, 2, 1).reshape(-1, self.in_channels, *self.roi_feat_size)
            cur_x = self.norm3(cur_x)
            x_list.append(cur_x)
        x = torch.cat(x_list)
        return super(TCTBBoxHead, self).forward(x)
