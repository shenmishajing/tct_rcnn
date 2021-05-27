import torch
import torchvision.transforms.functional as F

from mmdet.core import bbox2result
from mmdet.core.bbox import bbox_xyxy_to_lxtywh
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .inceptionv3 import inception_v3


@DETECTORS.register_module()
class YOLOV3Classifier(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 classifier = None,
                 hard_labels = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None,
                 init_cfg = None):
        super(YOLOV3Classifier, self).__init__(backbone, neck, bbox_head, train_cfg,
                                               test_cfg, pretrained, init_cfg)
        if classifier is not None:
            self.classifier = inception_v3(**classifier.pop('model'))
            self.classifier_cfg = classifier
            self.hard_labels = hard_labels
            self.hard_label_to_id = {label: i for i, label in enumerate(hard_labels)}

    @property
    def with_classifier(self):
        """bool: whether the detector has a classifier"""
        return hasattr(self, 'classifier') and self.classifier is not None and hasattr(self, 'hard_labels') and self.hard_labels is not None

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

        if self.with_classifier:
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
        raise NotImplementedError
