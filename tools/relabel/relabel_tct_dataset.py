import os
import os.path as osp
import numpy as np
import argparse
import json
import shutil
from collections import defaultdict

import mmcv


def calculate_area(bboxes):
    w = np.maximum(bboxes[..., 2] - bboxes[..., 0], 0)
    h = np.maximum(bboxes[..., 3] - bboxes[..., 1], 0)
    return w * h


def calculate_iou(res, gt):
    res_area = calculate_area(res)
    gt_area = calculate_area(gt)
    inter_bboxes = np.concatenate([np.maximum(res[:, None, :2], gt[None, :, :2]), np.minimum(res[:, None, 2:], gt[None, :, 2:])], axis = -1)
    inter_area = calculate_area(inter_bboxes)
    return inter_area / (res_area[:, None] + gt_area[None, :] - inter_area)


def write_anns(prediction_path, ann_dir, round = 2, score_thr = 0.5, iou_thr = 0.5, train_files = None):
    """write anns.

    Args:
        prediction_path (str): The dir of round of det results from test results pkl file
        ann_dir (str): The dir of ann files.
        round (int): The number of round to gen.
        score_thr (float, optional): The score threshold for bboxes.
        iou_thr (float, optional): The iou threshold for bboxes.
        train_files (Set[str], optional): The train file names.
    """
    prediction_path = os.path.join(prediction_path, f'round_{round - 1}')
    for dir_name in os.listdir(prediction_path):
        if not os.path.isdir(os.path.join(prediction_path, dir_name)):
            continue
        # relabel xml file
        result_path = os.path.join(prediction_path, dir_name, 'result.pkl')
        if not os.path.isfile(result_path):
            continue
        results = mmcv.load(result_path)
        src_path = os.path.join(ann_dir, f'round_{round - 1}', dir_name)
        des_path = os.path.join(ann_dir, f'round_{round}', dir_name)
        if os.path.exists(des_path):
            continue
        shutil.copytree(src_path, des_path)
        for path, dirs, files in os.walk(des_path):
            if 'train_normal.json' not in files:
                continue
            anns = json.load(open(osp.join(path, 'train_normal.json')))
            img_to_anns = defaultdict(lambda: defaultdict(list))
            for ann in anns['annotations']:
                img_to_anns[ann['image_id']][ann['category_id']].append(ann['bbox'])

            prog_bar = mmcv.ProgressBar(len(results))
            for i, result in enumerate(results):
                img_info = anns['images'][i]
                assert img_info['id'] == i, 'image id mismatch'
                if train_files is not None and img_info['filename'][:-4] not in train_files:
                    continue
                for c, res in enumerate(result):
                    res = res[res[:, -1] >= score_thr, :4]
                    gt = np.array(img_to_anns[i][c])
                    if len(gt) > 0:
                        gt[:, 2] += gt[:, 0]
                        gt[:, 3] += gt[:, 1]
                        iou = calculate_iou(res, gt)
                        iou = np.max(iou, axis = 1)
                        res = res[iou < iou_thr]
                    if not len(res):
                        continue
                    for bbox in res:
                        bbox = [int(b + 0.5) for b in bbox]
                        bbox_ann = {}
                        bbox_ann['segmentation'] = [[bbox[0], bbox[1], bbox[0], bbox[3], bbox[2], bbox[3], bbox[2], bbox[1]]]
                        bbox_ann['area'] = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                        bbox_ann['iscrowd'] = 0
                        bbox_ann['ignore'] = 0
                        bbox_ann['image_id'] = i
                        bbox_ann['bbox'] = [bbox[0], bbox[1], max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])]
                        bbox_ann['category_id'] = c
                        bbox_ann['id'] = len(anns['annotations'])
                        anns['annotations'].append(bbox_ann)
                prog_bar.update()
            json.dump(anns, open(osp.join(path, 'train_normal.json'), 'w'))


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'MMDet eval image prediction result for each')
    parser.add_argument(
        'prediction_path', help = 'prediction path where round folder exist')
    parser.add_argument(
        'ann_dir', help = 'directory where the old round of ann saved and new round of ann will be saved')
    parser.add_argument(
        '--round', type = int, default = None, help = 'the round of ann files to gen')
    parser.add_argument(
        '--score-thr',
        type = float,
        default = 0.5,
        help = 'score threshold (default: 0.5)')
    parser.add_argument(
        '--iou-thr',
        type = float,
        default = 0.5,
        help = 'iou threshold (default: 0.5)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.round is None:
        args.round = 5

    train_file_path = '/data/zhengwenhao/Datasets/TCTDataSet/ImageSets/Main/train.txt'
    train_files = set([line.strip() for line in open(train_file_path).readlines()])
    write_anns(args.prediction_path, args.ann_dir, args.round, args.score_thr, args.iou_thr, train_files)


if __name__ == '__main__':
    main()
