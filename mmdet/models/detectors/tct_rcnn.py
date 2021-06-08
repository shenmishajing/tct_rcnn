import torch
import numpy as np
from torch import nn
from torch.nn import Dropout2d
from dropblock import DropBlock2D

import mmcv
from mmdet.core.visualization import imshow_det_bboxes
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class TCTRCNN(TwoStageDetector):
    """Implementation of `TCT R-CNN`_"""

    def __init__(self,
                 backbone,
                 neck = None,
                 rpn_head = None,
                 noise_module = None,
                 roi_head = None,
                 part = 'abnormal',
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None,
                 init_cfg = None):
        super(TwoStageDetector, self).__init__(init_cfg)
        self.stages = ['normal', 'abnormal']
        self.part = part
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            if not isinstance(rpn_head, dict) or self.stages[0] not in rpn_head:
                rpn_head = {stage: rpn_head for stage in self.stages}
            self.rpn_head = nn.ModuleDict()
            for stage in self.stages:
                rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
                rpn_head_ = rpn_head[stage].copy()
                rpn_head_.update(train_cfg = rpn_train_cfg, test_cfg = test_cfg.rpn)
                self.rpn_head[stage] = build_head(rpn_head_)

        if noise_module is not None:
            noise_module_cls = eval(noise_module.pop('type'))
            self.noise_module = noise_module_cls(**noise_module)

        if roi_head is not None:
            if not isinstance(roi_head, dict) or self.stages[0] not in roi_head:
                roi_head = {stage: roi_head for stage in self.stages}
            self.roi_head = nn.ModuleDict()
            for stage in self.stages:
                # update train and test cfg here for now
                # TODO: refactor assigner & sampler
                roi_head_ = roi_head[stage].copy()
                if 'train_cfg' not in roi_head_:
                    roi_head_['train_cfg'] = train_cfg.rcnn if train_cfg is not None else None
                if 'test_cfg' not in roi_head_:
                    roi_head_['test_cfg'] = test_cfg.rcnn
                self.roi_head[stage] = build_head(roi_head_)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_noise(self):
        """bool: whether the detector has a noise module"""
        return hasattr(self, 'noise_module') and self.noise_module is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore = None,
                      gt_masks = None,
                      proposals = None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override normal det bboxes with custom proposals.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        ################
        # Normal stage
        ################

        # inference
        if proposals is None:
            with torch.no_grad():
                self.eval()
                proposal_list = self.rpn_head['normal'].simple_test_rpn(x, img_metas)
                det_bboxes, det_labels = self.roi_head['normal'].simple_test_bboxes(x, img_metas, proposal_list,
                                                                                    self.roi_head['normal'].test_cfg)
                det_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            self.train()

            normal_x = x
            normal_bboxes = kwargs['normal']['gt_bboxes']
            normal_labels = kwargs['normal']['gt_labels']
            # noise
            if self.with_noise:
                normal_x = tuple(self.noise_module(feat) for feat in normal_x)
                normal_bboxes = [torch.cat([det_bbox, gt_bbox]) for det_bbox, gt_bbox in zip(det_bboxes, normal_bboxes)]
                normal_labels = [torch.cat([det_label, gt_label]) for det_label, gt_label in zip(det_labels, normal_labels)]

            # RPN forward and loss
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head['normal'].forward_train(normal_x, img_metas, normal_bboxes, gt_labels = None,
                                                                              gt_bboxes_ignore = None, proposal_cfg = proposal_cfg)
            for k, v in rpn_losses.items():
                losses['normal_' + k] = v

            roi_losses = self.roi_head['normal'].forward_train(normal_x, img_metas, proposal_list, normal_bboxes, normal_labels, None, None,
                                                               **kwargs)
            for k, v in roi_losses.items():
                losses['normal_' + k] = v

            if self.part == 'normal':
                return losses
            det_bboxes = [bbox[label == 0] for bbox, label in zip(det_bboxes, det_labels)]
        else:
            det_bboxes = proposals

        ################
        # Abnormal stage
        ################

        abnormal_bboxes = kwargs['tct']['gt_bboxes']
        abnormal_labels = kwargs['tct']['gt_labels']

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head['abnormal'].forward_train(x, img_metas, abnormal_bboxes, gt_labels = None,
                                                                            gt_bboxes_ignore = None,
                                                                            proposal_cfg = proposal_cfg)
        losses.update(rpn_losses)

        roi_losses = self.roi_head['abnormal'].forward_train(x, img_metas, proposal_list, abnormal_bboxes, abnormal_labels, None, None,
                                                             det_bboxes, **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals = None,
                                rescale = False):
        """Async test without augmentation."""
        x = self.extract_feat(img)

        ################
        # Normal stage
        ################

        # inference
        if proposals is None:
            proposal_list = await self.rpn_head['normal'].async_simple_test_rpn(x, img_meta)

            if self.part == 'normal':
                return await self.roi_head['normal'].async_simple_test(x, proposal_list, img_meta, rescale = rescale)

            det_bboxes, det_labels = self.roi_head['normal'].simple_test_bboxes(x, img_meta, proposal_list,
                                                                                self.roi_head['normal'].test_cfg)
            det_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            det_bboxes = [bbox[label == 0] for bbox, label in zip(det_bboxes, det_labels)]
        else:
            det_bboxes = proposals

        ################
        # Abnormal stage
        ################

        # RPN forward and loss
        proposal_list = await self.rpn_head['abnormal'].async_simple_test_rpn(x, img_meta, rescale = rescale)
        return await self.roi_head['abnormal'].async_simple_test(x, proposal_list, img_meta, det_bboxes)

    def simple_test(self, img, img_metas, proposals = None, rescale = False):
        """Test without augmentation."""
        x = self.extract_feat(img)

        ################
        # Normal stage
        ################

        # inference
        if proposals is None:
            proposal_list = self.rpn_head['normal'].simple_test_rpn(x, img_metas)

            if self.part == 'normal':
                return self.roi_head['normal'].simple_test(x, proposal_list, img_metas, rescale = rescale)

            det_bboxes, det_labels = self.roi_head['normal'].simple_test_bboxes(x, img_metas, proposal_list,
                                                                                self.roi_head['normal'].test_cfg)
            det_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            det_bboxes = [bbox[label == 0] for bbox, label in zip(det_bboxes, det_labels)]
        else:
            det_bboxes = proposals

        ################
        # Abnormal stage
        ################

        # RPN forward and loss
        proposal_list = self.rpn_head['abnormal'].simple_test_rpn(x, img_metas)

        return self.roi_head['abnormal'].simple_test(x, proposal_list, img_metas, det_bboxes, rescale = rescale)

    def aug_test(self, imgs, img_metas, proposals = None, rescale = False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)

        ################
        # Normal stage
        ################

        # inference
        if proposals is None:
            proposal_list = self.rpn_head['normal'].aug_test_rpn(x, img_metas)
            if self.part == 'normal':
                return self.roi_head['normal'].aug_test(x, proposal_list, img_metas, rescale = rescale)

            det_bboxes, det_labels = self.roi_head['normal'].aug_test_bboxes(x, img_metas, proposal_list, self.roi_head['normal'].test_cfg)
            det_bboxes = [bbox[:, :4] for bbox in det_bboxes]
            det_bboxes = [bbox[label == 0] for bbox, label in zip(det_bboxes, det_labels)]
        else:
            det_bboxes = proposals

        ################
        # Abnormal stage
        ################

        # RPN forward and loss
        proposal_list = self.rpn_head['abnormal'].aug_test_rpn(x, img_metas)

        return self.roi_head['abnormal'].aug_test(x, proposal_list, img_metas, det_bboxes, rescale = rescale)
