import os.path as osp
import numpy as np
import argparse
import json
from collections import defaultdict

import mmcv


def calculate_iou(res, gt):
    pass


def evaluate_and_show(results, ann_dir, score_thr = 0.5, iou_thr = 0.5):
    """Evaluate and show results.

    Args:
        results (list): Det results from test results pkl file
        ann_dir (str): The dir of ann files.
        score_thr (float, optional): The score threshold for bboxes.
    """
    anns = json.load(open(osp.join(ann_dir, 'tct_normal.json')))
    img_to_anns = defaultdict(lambda: defaultdict(list))
    for ann in anns['annotations']:
        img_to_anns[ann['image_id']][ann['category_id']].append(ann['bbox'])

    prog_bar = mmcv.ProgressBar(len(results))
    for i, (result,) in enumerate(zip(results)):
        for c, res in enumerate(result):
            res = res[res[:, -1] >= score_thr]
            gt = np.array(img_to_anns[i][c])
            if len(gt) > 0:
                gt[:, 2] += gt[:, 0]
                gt[:, 3] += gt[:, 1]
                iou = calculate_iou(res, gt)
        img_info = anns['images'][i]
        assert img_info['id'] == i, 'image id mismatch'
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'MMDet eval image prediction result for each')
    parser.add_argument(
        'prediction_path', help = 'prediction path where test pkl result')
    parser.add_argument(
        'ann_dir', help = 'directory where ann saved and new ann will be saved')
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
    mmcv.check_file_exist(args.prediction_path)
    outputs = mmcv.load(args.prediction_path)
    evaluate_and_show(outputs, args.ann_dir, args.score_thr, args.iou_thr)


if __name__ == '__main__':
    main()
