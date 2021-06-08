import os
import json
from collections import defaultdict
import numpy as np
import colorsys
import cv2

import mmcv


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

# 坐标顺序： 上-》左-》下-》右
def draw_bounding_box_on_image(img,
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
    (str_width, str_height), base_line = cv2.getTextSize(display_str, fontFace = font_face, fontScale = font_scale, thickness = thickness)
    # 计算覆盖文本的矩形框坐标
    cv2.rectangle(img, (left, top - str_height - 4), (left + str_width, top), thickness = -1, color = color)
    # 绘制文本
    cv2.putText(img, display_str, (left, top - 2), fontScale = font_scale, fontFace = font_face, thickness = thickness,
                color = (256, 256, 256))


def draw_image(image, image_id_to_anns, images_path, image_output_path):
    if not len(image_id_to_anns[image['id']]):
        return
    img = cv2.imread(os.path.join(images_path, image['filename']))
    for bbox in image_id_to_anns[image['id']]:
        draw_bounding_box_on_image(img, *bbox['bbox'], color = bbox['color'], display_str = bbox['name'])
    cv2.imwrite(os.path.join(image_output_path, image['filename']), img)


def draw_images(json_path, images_path, image_output_path):
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    ann_dict = json.load(open(json_path))
    # colors = get_colors(len(ann_dict['categories']))
    # category_dict = {category['id']: {'name': category['name'], 'color': color} for category, color in zip(ann_dict['categories'], colors)}
    colors = get_colors(6)
    category_dict = {category['id']: {
        'name': category['name'] if 'multi' in category['name'] else category['name'] + '-single',
        'color': colors[i] if i < 6 else colors[i - 5]} for i, category in enumerate(ann_dict['categories'])}
    image_id_to_anns = defaultdict(list)
    for ann in ann_dict['annotations']:
        if category_dict[ann['category_id']]['name'] == 'NORMAL-single':
            continue
        image_id_to_anns[ann['image_id']].append(
            {'bbox': [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],
             'name': category_dict[ann['category_id']]['name'],
             'color': category_dict[ann['category_id']]['color']})
    prog_bar = mmcv.ProgressBar(len(ann_dict['images']))
    for image in ann_dict['images']:
        draw_image(image, image_id_to_anns, images_path, image_output_path)
        prog_bar.update()


def main():
    data_dir = "/data/zhengwenhao/Datasets/TCTDataSet"
    json_path = os.path.join(data_dir, 'coco/tct_all.json')
    images_path = os.path.join(data_dir, 'JPEGImages')
    output_path = os.path.join(data_dir, 'middle_results/outputs')
    image_output_path = 'images'

    image_output_path = os.path.join(output_path, image_output_path)
    draw_images(json_path, images_path, image_output_path)


if __name__ == '__main__':
    main()
