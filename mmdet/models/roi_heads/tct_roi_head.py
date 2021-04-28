import torch
import torch.nn as nn
import random

from mmdet.core import bbox2result, bbox2roi, bbox_mapping, build_assigner, build_sampler, merge_aug_bboxes, merge_aug_masks, multiclass_nms
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead
from .bbox_heads.convfc_relation_bbox_head import ConvFCRelationBBoxHead


@HEADS.register_module()
class TCTRoIHead(CascadeRoIHead):
    """RoI head for TCT RCNN.
    """

    def __init__(self, num_classes, stage_loss_weights, train_cfg, num_memory = 10, *args, **kwargs):
        self.stages = ['single', 'multi', 'tct']
        if stage_loss_weights is None or stage_loss_weights == {} or self.stages[0] not in self.stages:
            stage_loss_weights = {}
            for stage in self.stages:
                stage_loss_weights[stage] = 1.0
        else:
            for stage in self.stages:
                assert stage in stage_loss_weights, f'can not find stage {stage} in stage_loss_weights'

        if not isinstance(train_cfg, dict) or self.stages[0] not in train_cfg:
            train_cfg = {stage: train_cfg for stage in self.stages}
        else:
            for stage in self.stages:
                assert stage in train_cfg, f'can not find stage {stage} in train_cfg'
        super(TCTRoIHead, self).__init__(num_stages = len(self.stages), stage_loss_weights = stage_loss_weights, train_cfg = train_cfg,
                                         *args, **kwargs)
        self.num_classes = num_classes
        self.memory_bank = []
        self.num_memory = num_memory

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = nn.ModuleDict()
        if not isinstance(bbox_head, dict) or self.stages[0] not in bbox_head:
            bbox_head = {stage: bbox_head for stage in self.stages}
        else:
            for stage in self.stages:
                assert stage in bbox_head, f'can not find stage {stage} in bbox_head'
        for stage in self.stages:
            self.bbox_head[stage] = build_head(bbox_head[stage])

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = dict()
        self.bbox_sampler = dict()
        if self.train_cfg is not None:
            for stage in self.stages:
                self.bbox_assigner[stage] = build_assigner(self.train_cfg[stage].assigner)
                self.bbox_sampler[stage] = build_sampler(self.train_cfg[stage].sampler)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained = pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            for stage in self.stages:
                self.bbox_head[stage].init_weights()

    def get_memory(self, tensor = None, device = torch.device('cpu'), dtype = torch.float32):
        if self.memory_bank:
            return random.choice(self.memory_bank)
        else:
            out_channels = self.bbox_roi_extractor.out_channels
            out_size = self.bbox_roi_extractor.roi_layers[0].output_size
            if tensor is not None:
                return torch.zeros(out_channels, *out_size, device = tensor.device, dtype = tensor.dtype)
            else:
                return torch.zeros(out_channels, *out_size, device = device, dtype = dtype)

    def update_memory(self, feats):
        if self.num_memory <= len(feats):
            inds = random.sample(range(len(feats)), self.num_memory)
            self.memory_bank = [feats[ind].clone().detach() for ind in inds]
        else:
            self.memory_bank = self.memory_bank[len(feats) - self.num_memory:] + [feat.clone().detach() for feat in feats]

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for stage in self.stages:
                bbox_results = self._bbox_forward(stage, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        return outs

    def _normal_forward(self, x, det_bboxes):
        # build normal_bbox_feats
        normal_rois = bbox2roi(det_bboxes)
        normal_bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], normal_rois)
        if len(normal_bbox_feats):
            self.update_memory(normal_bbox_feats)
            bbox_feats = []
            for i in range(len(det_bboxes)):
                cur_bbox_feats = normal_bbox_feats[normal_rois[:, 0] == i]
                if len(cur_bbox_feats):
                    bbox_feats.append(random.choice(cur_bbox_feats))
                else:
                    bbox_feats.append(None)
        else:
            bbox_feats = None

        return bbox_feats

    def _bbox_forward(self, stage, x, rois, normal_bbox_feats = None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        feats = []
        inds = rois[:, 0].unique()
        for i in inds:
            i = int(i)
            cur_bbox_feats = bbox_feats[rois[:, 0] == i]
            if len(cur_bbox_feats):
                if normal_bbox_feats is None or normal_bbox_feats[i] is None:
                    cur_normal_bbox_feats = self.get_memory(cur_bbox_feats[0])
                else:
                    cur_normal_bbox_feats = normal_bbox_feats[i]
                cur_normal_bbox_feats = cur_normal_bbox_feats[None, ...].expand_as(cur_bbox_feats)
                cur_bbox_feats = 2 * cur_bbox_feats - cur_normal_bbox_feats
                feats.append(cur_bbox_feats)
        bbox_feats = torch.cat(feats)
        # do not support caffe_c4 model anymore
        if isinstance(bbox_head, ConvFCRelationBBoxHead):
            cls_score, bbox_pred = bbox_head(bbox_feats, rois)
        else:
            cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score = cls_score, bbox_pred = bbox_pred, bbox_feats = bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg, normal_bbox_feats = None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois, normal_bbox_feats)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox = loss_bbox, rois = rois, bbox_targets = bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore = None,
                      gt_masks = None,
                      det_bboxes = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        final_proposal_list = []
        normal_bbox_feats = self._normal_forward(x, det_bboxes)

        # head inference
        for stage in self.stages:
            lw = self.stage_loss_weights[stage]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                if stage == self.stages[-1]:
                    cur_proposal_list = [torch.cat([proposal[j] for proposal in final_proposal_list]) for j in range(num_imgs)]
                    cur_gt_bboxes = gt_bboxes
                    cur_gt_labels = gt_labels
                    cur_gt_bboxes_ignore = gt_bboxes_ignore
                else:
                    cur_proposal_list = proposal_list
                    cur_gt_bboxes = kwargs[stage]['gt_bboxes']
                    cur_gt_labels = kwargs[stage]['gt_labels']
                    cur_gt_bboxes_ignore = kwargs[stage].get('gt_bboxes_ignore', None)
                    if cur_gt_bboxes_ignore is None:
                        cur_gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(cur_proposal_list[j], cur_gt_bboxes[j], cur_gt_bboxes_ignore[j], cur_gt_labels[j])
                    sampling_result = bbox_sampler.sample(assign_result, cur_proposal_list[j], cur_gt_bboxes[j], cur_gt_labels[j],
                                                          feats = [lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

                # bbox head forward and loss
                bbox_results = self._bbox_forward_train(stage, x, sampling_results, cur_gt_bboxes, cur_gt_labels, self.train_cfg[stage],
                                                        normal_bbox_feats)

                for name, value in bbox_results['loss_bbox'].items():
                    losses[f'{stage}_{name}'] = (
                        value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    stage, x, sampling_results, gt_masks, self.train_cfg[stage],
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f'{stage}_{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if stage != self.stages[-1]:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[stage].num_classes,
                        bbox_results['cls_score'][:, :-1].argmax(1),
                        roi_labels)
                    final_proposal_list.append(
                        self.bbox_head[stage].refine_bboxes(bbox_results['rois'], roi_labels, bbox_results['bbox_pred'],
                                                            pos_is_gts, img_metas))

        return losses

    def simple_test(self, x, proposal_list, img_metas, det_bboxes, rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        final_proposal_list = []
        normal_bbox_feats = self._normal_forward(x, det_bboxes)
        for stage in self.stages:
            if stage == self.stages[-1]:
                cur_rois = torch.cat(final_proposal_list)
                proposal_list = []
                for i in range(len(img_metas)):
                    cur_inds = cur_rois[:, 0] == i
                    proposal_list.append(cur_rois[cur_inds])
                cur_rois = torch.cat(proposal_list)
            else:
                cur_rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, cur_rois, normal_bbox_feats)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            cur_rois = cur_rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[stage].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if stage != self.stages[-1]:
                bbox_label = [s[:, :-1].argmax(dim = 1) for s in cls_score]
                final_proposal_list.append(torch.cat([
                    self.bbox_head[stage].regress_by_class(cur_rois[j], bbox_label[j], bbox_pred[j], img_metas[j]) for j in range(num_imgs)
                ]))

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[self.stages[-1]].get_bboxes(
                cur_rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale = rescale,
                cfg = rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[self.stages[-1]].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # split batch mask prediction back to each image
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append(
                        [m.sigmoid().cpu().numpy() for m in mask_pred])

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale = False):
        """Test with augmentations.
        """
        raise NotImplementedError('not supported yet')
