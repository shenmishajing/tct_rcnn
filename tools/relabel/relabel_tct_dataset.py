import os
import os.path as osp
import numpy as np
import argparse
import json
import shutil
import xmltodict
import requests
from collections import defaultdict, OrderedDict

import mmcv
from VOC2COCO import gen_coco
from tct_class_gen import gen_tct_class_annotations
from analysis import draw_images


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


def write_anns(results, ann_dir, score_thr = 0.5, iou_thr = 0.5):
    """write anns.

    Args:
        results (list): Det results from test results pkl file
        ann_dir (str): The dir of ann files.
        score_thr (float, optional): The score threshold for bboxes.
    """
    anns = json.load(open(osp.join(ann_dir, 'coco/tct_normal.json')))
    img_to_anns = defaultdict(lambda: defaultdict(list))
    for ann in anns['annotations']:
        img_to_anns[ann['image_id']][ann['category_id']].append(ann['bbox'])

    prog_bar = mmcv.ProgressBar(len(results))
    for i, (result,) in enumerate(zip(results)):
        img_info = anns['images'][i]
        assert img_info['id'] == i, 'image id mismatch'
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
            cur_ann_file = osp.join(ann_dir, 'Annotations', osp.splitext(img_info['filename'])[0] + '.xml')
            xml_ann = xmltodict.parse(open(cur_ann_file).read())
            for bbox in res:
                bbox_ann = OrderedDict()
                bbox_ann['name'] = anns['categories'][c]['name']
                bbox_ann['pose'] = 'Unspecified'
                bbox_ann['truncated'] = '0'
                bbox_ann['difficult'] = '0'
                bbox_ann['bndbox'] = OrderedDict()
                bbox_ann['bndbox']['xmin'] = str(int(bbox[0]))
                bbox_ann['bndbox']['ymin'] = str(int(bbox[1]))
                bbox_ann['bndbox']['xmax'] = str(int(bbox[2]))
                bbox_ann['bndbox']['ymax'] = str(int(bbox[3]))
                if 'object' not in xml_ann['annotation']:
                    xml_ann['annotation']['object'] = bbox_ann
                elif isinstance(xml_ann['annotation']['object'], list):
                    xml_ann['annotation']['object'].append(bbox_ann)
                elif isinstance(xml_ann['annotation']['object'], OrderedDict):
                    xml_ann['annotation']['object'] = [xml_ann['annotation']['object'], bbox_ann]
                else:
                    print(xml_ann)
            xml_str = xmltodict.unparse(xml_ann)
            url = "http://web.chacuo.net/formatxml"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
                "Host": "web.chacuo.net",
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            }
            form_data = {"data": xml_str, "type": "format", "beforeSend": "undefined"}
            try:
                resp = requests.post(url, data = form_data, headers = headers, timeout = 20)
                xml_str = resp.json()['data'][0]
            except Exception as e:
                print(e)
            xml_file = open(cur_ann_file, 'w')
            xml_file.write(xml_str)
            xml_file.close()
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

    # relabel xml file
    mmcv.check_file_exist(args.prediction_path)
    outputs = mmcv.load(args.prediction_path)
    write_anns(outputs, args.ann_dir, args.score_thr, args.iou_thr)

    # remove coco ann files
    if osp.exists(osp.join(args.ann_dir, 'coco')):
        shutil.rmtree(osp.join(args.ann_dir, 'coco'))
        os.mkdir(osp.join(args.ann_dir, 'coco'))

    # gen coco main file
    gen_coco(args.ann_dir, osp.join(args.ann_dir, 'coco'), osp.join(args.ann_dir, 'obj.names'))

    # gen all coco ann files
    gen_tct_class_annotations(osp.join(args.ann_dir, 'coco'))

    normal_image_ouput_path = osp.join(args.ann_dir, 'middle_results/outputs/images')
    # remove normal bboxes images file
    if osp.exists(normal_image_ouput_path):
        shutil.rmtree(normal_image_ouput_path)
        os.makedirs(normal_image_ouput_path)

    # draw normal bboxes
    draw_images(osp.join(args.ann_dir, 'coco/tct_normal.json'), osp.join(args.ann_dir, 'JPEGImages'), normal_image_ouput_path)


if __name__ == '__main__':
    main()
