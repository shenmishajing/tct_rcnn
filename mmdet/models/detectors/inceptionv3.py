import logging

import torch
import torch.nn as nn
from torchvision.models.inception import Inception3
from torchvision.models.utils import load_state_dict_from_url

from mmcv.runner import load_checkpoint

from ..builder import DETECTORS
from .base import BaseDetector

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
class InceptionV3(BaseDetector):

    def __init__(self,
                 num_classes = 1000,
                 pretrained = None,
                 **kwargs):
        super(InceptionV3, self).__init__()
        self.model = inception_v3(num_classes = num_classes, pretrained = pretrained)
        self.loss = nn.CrossEntropyLoss()

    def extract_feat(self, imgs):
        """Extract features from images."""
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(InceptionV3, self).forward_train(img, img_metas)
        out = self.model(img)
        if not isinstance(out, torch.Tensor):
            out = out.logits
        return {'loss_cls': self.loss(out, torch.cat(gt_labels))}

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

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
        outs = self.model(img)
        if not isinstance(outs, torch.Tensor):
            outs = outs.logits
        return [torch.argmax(out) for out in outs]

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError

    def show_result(self, **kwargs):
        pass
