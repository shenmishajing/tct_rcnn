import torch
import torch.nn as nn

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class ComparisonDetector(TwoStageDetector):
    """Implementation of `Comparison detector for cervical cell/clumps detection in the limited data scenario
     <https://www.sciencedirect.com/science/article/pii/S092523122100014X?via=ihub>`"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck = None,
                 pretrained = None):
        super(ComparisonDetector, self).__init__(
            backbone = backbone,
            neck = neck,
            rpn_head = rpn_head,
            roi_head = roi_head,
            train_cfg = train_cfg,
            test_cfg = test_cfg,
            pretrained = pretrained)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.conv = nn.Conv2d(256 * 5, 256, 1)

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

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        reference = []
        for references in kwargs['references']:
            cur_ref = [self.extract_feat(ref['img'][0, None]) for ref in references]
            cur_ref = [self.pool(torch.mean(torch.cat([ref[i] for ref in cur_ref]), dim = 0)) for i in range(len(cur_ref[0]))]
            reference.append(torch.mean(torch.stack(cur_ref), dim = 0))
        reference = torch.stack(reference)
        background_reference = self.conv(reference.reshape(1, -1, 7, 7))
        reference = torch.cat([reference, background_reference])

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels = None,
                gt_bboxes_ignore = gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, reference, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals = None,
                                rescale = False,
                                **kwargs):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        reference = []
        for references in kwargs['references']:
            cur_ref = [self.extract_feat(ref['img'][0, None]) for ref in references]
            cur_ref = [self.pool(torch.mean(torch.cat([ref[i] for ref in cur_ref]), dim = 0)) for i in range(len(cur_ref[0]))]
            reference.append(torch.mean(torch.stack(cur_ref), dim = 0))
        reference = torch.stack(reference)
        background_reference = self.conv(reference.reshape(1, -1, 7, 7))
        reference = torch.cat([reference, background_reference])

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, reference, img_meta, rescale = rescale)

    def simple_test(self, img, img_metas, proposals = None, rescale = False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        reference = []
        for references in kwargs['references']:
            cur_ref = [self.extract_feat(ref['img'][0, None]) for ref in references]
            cur_ref = [self.pool(torch.mean(torch.cat([ref[i] for ref in cur_ref]), dim = 0)) for i in range(len(cur_ref[0]))]
            reference.append(torch.mean(torch.stack(cur_ref), dim = 0))
        reference = torch.stack(reference)
        background_reference = self.conv(reference.reshape(1, -1, 7, 7))
        reference = torch.cat([reference, background_reference])

        x = self.extract_feat(img)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, reference, rescale = rescale)

    def aug_test(self, imgs, img_metas, rescale = False, **kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        reference = []
        for references in kwargs['references']:
            cur_ref = [self.extract_feat(ref['img'][0, None]) for ref in references]
            cur_ref = [self.pool(torch.mean(torch.cat([ref[i] for ref in cur_ref]), dim = 0)) for i in range(len(cur_ref[0]))]
            reference.append(torch.mean(torch.stack(cur_ref), dim = 0))
        reference = torch.stack(reference)
        background_reference = self.conv(reference.reshape(1, -1, 7, 7))
        reference = torch.cat([reference, background_reference])

        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, reference, rescale = rescale)
