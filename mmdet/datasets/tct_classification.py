import logging
import os.path as osp
import numpy as np
from .api_wrappers import COCO
from .builder import DATASETS
from .pipelines import Compose
from .coco import CocoDataset


@DATASETS.register_module()
class TCTClassificationDataset(CocoDataset):
    CLASSES = ['ASCH', 'ASCUS', 'HSIL', 'LSIL']

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root = None,
                 img_prefix = '',
                 seg_prefix = None,
                 proposal_file = None,
                 test_mode = False,
                 filter_min_size = 32,
                 filter_empty_gt = True):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.filter_min_size = filter_min_size

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs(self.filter_min_size)
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    # def __len__(self):
    #     """Total number of samples of data."""
    #     return 4

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str):  Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names = self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.ann_ids = self.coco.get_ann_ids()
        self.ann_id_to_img = {}
        data_infos = []
        for i in self.ann_ids:
            info = self.coco.load_anns(i)[0]
            data_infos.append(info)
            self.ann_id_to_img[i] = self.coco.load_imgs(info['image_id'])[0]
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ann_info = self.data_infos[idx]
        img_info = self.ann_id_to_img[ann_info['id']]
        return img_info, self._parse_ann_info(img_info, [ann_info])

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        ann_info = self.data_infos[idx]
        return [ann_info['category_id']]

    def _filter_imgs(self, min_size = 32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.coco.cat_img_map[class_id])

        for i, ann_info in enumerate(self.data_infos):
            ann_id = self.ann_ids[i]
            img_info = self.ann_id_to_img[ann_id]
            img_id = img_info['id']
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype = np.uint8)
        for i in range(len(self)):
            ann_info = self.data_infos[i]
            img_info = self.ann_id_to_img[ann_info['id']]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info, ann_info = self.get_ann_info(idx)
        results = dict(img_info = img_info, ann_info = ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info, ann_info = self.get_ann_info(idx)
        results = dict(img_info = img_info, ann_info = ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 logger = None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        correct = 0
        for i in range(len(self)):
            gt_label = self.cat2label[self.data_infos[i]['category_id']]
            pre_label = int(results[i])
            if pre_label == gt_label:
                correct += 1
        return {'acc': correct / len(self)}
