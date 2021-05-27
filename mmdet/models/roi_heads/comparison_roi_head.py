import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class ComparisonRoIHead(StandardRoIHead):
    """Comparison roi head."""

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      reference,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore = None,
                      gt_masks = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
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
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats = [lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, reference, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, reference, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, reference)

        bbox_results = dict(
            cls_score = cls_score, bbox_pred = bbox_pred, bbox_feats = bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, reference, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, reference, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox = loss_bbox)
        return bbox_results

    async def async_simple_test(self, *args, **kwargs):
        """Async test without augmentation."""
        raise NotImplementedError

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           reference,
                           rcnn_test_cfg,
                           rescale = False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor or List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # The length of proposals of different batches may be different.
        # In order to form a batch, a padding operation is required.
        if isinstance(proposals, list):
            # padding to form a batch
            max_size = max([proposal.size(0) for proposal in proposals])
            for i, proposal in enumerate(proposals):
                supplement = proposal.new_full(
                    (max_size - proposal.size(0), proposal.size(1)), 0)
                proposals[i] = torch.cat((supplement, proposal), dim = 0)
            rois = torch.stack(proposals, dim = 0)
        else:
            rois = proposals

        batch_index = torch.arange(
            rois.size(0), device = rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim = -1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, reference, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img, -1)

        if not torch.onnx.is_in_onnx_export():
            # remove padding
            supplement_mask = rois[..., -1] == 0
            cls_score[supplement_mask, :] = 0

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.reshape(batch_size,
                                              num_proposals_per_img, -1)
                if not torch.onnx.is_in_onnx_export():
                    bbox_pred[supplement_mask, :] = 0
            else:
                # TODO: Looking forward to a better way
                # For SABL
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
                # apply bbox post-processing to each image individually
                det_bboxes = []
                det_labels = []
                for i in range(len(proposals)):
                    # remove padding
                    supplement_mask = proposals[i][..., -1] == 0
                    for bbox in bbox_preds[i]:
                        bbox[supplement_mask] = 0
                    det_bbox, det_label = self.bbox_head.get_bboxes(
                        rois[i],
                        cls_score[i],
                        bbox_preds[i],
                        img_shapes[i],
                        scale_factors[i],
                        rescale = rescale,
                        cfg = rcnn_test_cfg)
                    det_bboxes.append(det_bbox)
                    det_labels.append(det_label)
                return det_bboxes, det_labels
        else:
            bbox_pred = None

        return self.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shapes,
            scale_factors,
            rescale = rescale,
            cfg = rcnn_test_cfg)

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    reference,
                    proposals = None,
                    rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, reference, self.test_cfg, rescale = rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale = rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale = rescale)
            return list(zip(bbox_results, segm_results))

    async def aug_test(self, *args, **kwargs):
        """Test with augmentations."""
        raise NotImplementedError
