import argparse
import os
import json
from collections import defaultdict
import numpy as np
import colorsys
import cv2

import mmcv
from mmcv import Config, DictAction

from mmdet.datasets import build_dataset, get_loading_pipeline


def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        color = tuple(int(c * 256) for c in color)
        colors.append((color[2], color[1], color[0]))
    return colors


def get_color(name):
    color_dict = {
        'LSIL': (15, 58, 205),
        'ASCH': (241, 177, 67),
        'HSIL': (0, 241, 241),
        'SQCA': (243, 29, 199),
        'ASCUS': (113, 244, 56)
    }
    for color_name, color in color_dict.items():
        if color_name in name:
            return color[2], color[1], color[0]
    return 0, 0, 0


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, categories, score_thr = 0):
        self.categories = categories
        self.score_thr = score_thr

    # 坐标顺序： 上-》左-》下-》右
    def draw_bounding_box_on_image(self,
                                   img,
                                   xmin,
                                   ymin,
                                   xmax,
                                   ymax,
                                   color = 'red',
                                   thickness = 2,
                                   display_str = ''):
        """
        Args:
          img: a cv2 img.
          ymin: ymin of bounding box.
          xmin: xmin of bounding box.
          ymax: ymax of bounding box.
          xmax: xmax of bounding box.
          color: color to draw bounding box. Default is red.
          thickness: line thickness. Default value is 4.
          display_str: string to display.
        """
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

        # 绘制Box框
        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1

        # 计算文本的宽高，baseLine
        (str_width, str_height), base_line = cv2.getTextSize(display_str, fontFace = font_face, fontScale = font_scale,
                                                             thickness = thickness)
        # 计算覆盖文本的矩形框坐标
        cv2.rectangle(img, (left, top - str_height - 4), (left + str_width, top), color, -1)
        # 绘制文本
        cv2.putText(img, display_str, (left, top - 2), fontScale = font_scale, fontFace = font_face, thickness = thickness,
                    color = (256, 256, 256))

    def _save_image_results(self, data_info, result = None, out_path = None):
        img = cv2.imread(data_info['filename'])
        if result is None:
            for label, bbox in zip(data_info['ann_info']['labels'], data_info['ann_info']['bboxes']):
                self.draw_bounding_box_on_image(img, *[int(b) for b in bbox], color = self.categories[label]['color'],
                                                display_str = self.categories[label]['name'])
        else:
            for label, bboxes in enumerate(result):
                for bbox in bboxes:
                    if bbox[-1] >= self.score_thr:
                        self.draw_bounding_box_on_image(img, *[int(b) for b in bbox[:4]], color = self.categories[label]['color'],
                                                        display_str = self.categories[label]['name'] + f'|{bbox[-1]:.2f}')
        cv2.imwrite(out_path, img)

    def show_result(self,
                    dataset,
                    results,
                    show_dir = 'work_dir',
                    prediction_name = ''):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
        """
        mmcv.mkdir_or_exist(show_dir)

        if results is None:
            number = len(dataset)
        else:
            number = len(results)
        if prediction_name:
            prediction_name = '_' + prediction_name
        prog_bar = mmcv.ProgressBar(number)
        for i in range(number):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i)
            file_name, ext = os.path.splitext(data_info['img_info']['filename'])
            out_path = os.path.join(show_dir, file_name + prediction_name + ext)
            self._save_image_results(data_info, None if results is None else results[i], out_path)
            prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'MMDet eval image prediction result for each')
    parser.add_argument('config', help = 'test config file path')
    parser.add_argument(
        '--show_dir', default = None, help = 'directory where painted images will be saved')
    parser.add_argument(
        '--prediction_path', default = None, help = 'prediction path where test pkl result')
    parser.add_argument(
        '--show-score-thr',
        type = float,
        default = 0,
        help = 'score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs = '+',
        action = DictAction,
        help = 'override some settings in the used config, the key-value pair '
               'in xxx=yyy format will be merged into config file. If the value to '
               'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
               'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
               'Note that the quotation marks are necessary and that no white space '
               'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    categories = []
    for label, category_id in enumerate(dataset.cat_ids[dataset.part]):
        category_item = {'label': label, 'name': dataset.CLASSES[label], 'color': get_color(dataset.CLASSES[label])}
        categories.append(category_item)
    result_visualizer = ResultVisualizer(categories, args.show_score_thr)
    if args.prediction_path is None or os.path.isfile(args.prediction_path):
        if args.prediction_path is not None:
            outputs = mmcv.load(args.prediction_path)
        else:
            outputs = None
        if args.show_dir is None:
            assert args.prediction_path is not None
            out_dir = os.path.join(os.path.dirname(args.prediction_path), os.path.splitext(os.path.split(args.prediction_path)[1])[0])
        else:
            out_dir = args.show_dir
        result_visualizer.show_result(dataset, outputs, out_dir)
    else:
        for prediction_path in os.listdir(args.prediction_path):
            if os.path.isfile(os.path.join(args.prediction_path, prediction_path)) and prediction_path.endswith('.pkl'):
                outputs = mmcv.load(os.path.join(args.prediction_path, prediction_path))
                if args.show_dir is None:
                    assert args.prediction_path is not None
                    out_dir = os.path.join(args.prediction_path, 'show_dir', os.path.splitext(prediction_path)[0])
                else:
                    out_dir = os.path.join(args.show_dir, os.path.splitext(prediction_path)[0])
                if os.path.exists(out_dir):
                    print(f'output dir {out_dir} exists, skip prediction {os.path.join(args.prediction_path, prediction_path)}')
                    continue
                prediction_name = os.path.splitext(prediction_path)[0]
                print(f'prediction_path: {os.path.join(args.prediction_path, prediction_path)}, '
                      f'out_dir: {out_dir}, prediction_name: {prediction_name}')
                result_visualizer.show_result(dataset, outputs, out_dir, prediction_name)

        if args.show_dir is None:
            assert args.prediction_path is not None
            out_dir = os.path.join(args.prediction_path, 'show_dir', 'ground_truth')
        else:
            out_dir = os.path.join(args.show_dir, 'ground_truth')
        if os.path.exists(out_dir):
            print(f'output dir {out_dir} exists, skip ground truth')
        else:
            result_visualizer.show_result(dataset, None, out_dir, 'ground_truth')


if __name__ == '__main__':
    main()
