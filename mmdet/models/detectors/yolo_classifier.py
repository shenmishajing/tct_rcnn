import logging

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.inception import Inception3
from torchvision.models.utils import load_state_dict_from_url

from mmdet.core import bbox2result
from mmdet.core.bbox import bbox_xyxy_to_lxtywh
from mmcv.runner import load_checkpoint
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained = False, progress = True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool or str): If True, returns a model pre-trained on ImageNet.
            If str, returns a model as weight file specified by str
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        kwargs['init_weights'] = False  # we are loading weights from a pretrained model
        model = Inception3(**kwargs)
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(model, pretrained, strict = False, logger = logger)
        else:
            state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                                  progress = progress)
            cur_state_dict = model.state_dict()
            for key in list(state_dict):
                if key not in cur_state_dict or state_dict[key].shape != cur_state_dict[key].shape:
                    del state_dict[key]
            model.load_state_dict(state_dict, strict = False)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return Inception3(**kwargs)


@DETECTORS.register_module()
class YOLOV3Classifier(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 hard_labels,
                 classifier = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None):
        super(YOLOV3Classifier, self).__init__(backbone, neck, bbox_head, train_cfg,
                                               test_cfg, pretrained)
        if classifier is not None:
            self.classifier = inception_v3(**classifier.pop('model'))
            self.classifier_cfg = classifier
        self.hard_labels = hard_labels
        self.hard_label_to_id = {label: i for i, label in enumerate(hard_labels)}

    @property
    def with_classifier(self):
        """bool: whether the detector has a classifier"""
        return hasattr(self, 'classifier') and self.classifier is not None

    def simple_test(self, img, img_metas, rescale = False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale = rescale)

        if self.classifier:
            for det_bboxes, det_labels in bbox_list:
                inds = torch.zeros_like(det_labels) > 0
                for label in self.hard_labels:
                    inds = inds | (det_labels == label)
                for i in range(len(det_labels)):
                    if not inds[i]:
                        continue
                    hard_bbox = bbox_xyxy_to_lxtywh(det_bboxes[i, :4])
                    hard_bbox = [hard_bbox[1], hard_bbox[0], hard_bbox[3], hard_bbox[2]]
                    hard_bbox = [int(x + 0.5) for x in hard_bbox]
                    cur_img = F.crop(img, *[int(x + 0.5) for x in hard_bbox])
                    if cur_img.numel() <= 0:
                        continue
                    cur_img = F.resize(cur_img, **self.classifier_cfg)
                    output = self.classifier(cur_img)
                    label = self.hard_labels[int(torch.argmax(output))]
                    det_labels[i] = label

        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale = False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale = rescale)]
