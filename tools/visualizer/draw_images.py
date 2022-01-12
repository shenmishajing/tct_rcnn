import os
import json
import shutil
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


# 坐标顺序： 上-》左-》下-》右
def draw_bounding_box_on_image(img,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               color = 'red',
                               direction = '',
                               thickness = 3,
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
    font_scale = 1.5

    # 计算文本的宽高，baseLine
    (str_width, str_height), base_line = cv2.getTextSize(display_str, fontFace = font_face, fontScale = font_scale, thickness = thickness)
    # 计算覆盖文本的矩形框坐标
    if 'bottom' not in direction:
        display_top = top - str_height - 4
    else:
        display_top = bottom + 2
    if 'right' not in direction:
        display_left = left
    else:
        display_left = right - str_width
    # 绘制文本框
    cv2.rectangle(img, (display_left, display_top), (display_left + str_width, display_top + str_height + 4), thickness = -1, color = color)
    # 绘制文本
    cv2.putText(img, display_str, (display_left, display_top + str_height + 2), fontScale = font_scale, fontFace = font_face,
                thickness = thickness, color = (256, 256, 256))


def draw_image(image, anns, images_path, image_output_path):
    if not len(anns):
        return
    img = cv2.imread(os.path.join(images_path, image['filename']))
    for bbox in anns:
        draw_bounding_box_on_image(img, *bbox['bbox'], color = bbox['color'], direction = bbox['direction'], display_str = bbox['name'])
    cv2.imwrite(os.path.join(image_output_path, image['filename']), img)


def draw_images(json_path, images_path, image_output_path):
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    ann_dict = json.load(open(json_path))
    colors = get_colors(len(ann_dict['categories']))
    category_dict = {category['id']: {'name': category['name'], 'color': color} for category, color in zip(ann_dict['categories'], colors)}
    # category_dict = {category['id']: {
    #     'name': category['name'] if 'multi' in category['name'] else category['name'] + '-single',
    #     'color': get_color(category['name'])} for i, category in enumerate(ann_dict['categories']) if category['name'] != 'NORMAL'}
    image_id_to_anns = defaultdict(list)
    for ann in ann_dict['annotations']:
        if ann['category_id'] not in category_dict:
            continue
        image_id_to_anns[ann['image_id']].append(
            {'bbox': [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],
             'name': category_dict[ann['category_id']]['name'],
             'color': category_dict[ann['category_id']]['color']})
    prog_bar = mmcv.ProgressBar(len(ann_dict['images']))
    for image in ann_dict['images']:
        draw_image(image, image_id_to_anns[image['id']], images_path, image_output_path)
        prog_bar.update()


def temp_main():
    data_dir = "/data/zhengwenhao/Datasets/object_detection/TCTDataSet"
    json_path = os.path.join(data_dir, 'coco/annotations/tct.json')
    images_path = os.path.join(data_dir, 'voc/JPEGImages')
    output_path = os.path.join(data_dir, 'middle_results/outputs')
    image_output_path = 'final_images'

    image_output_path = os.path.join(output_path, image_output_path)
    if os.path.exists(image_output_path):
        shutil.rmtree(image_output_path)
    os.makedirs(image_output_path)

    ann_dict = json.load(open(json_path))
    category_dict = {category['id']: {'name': category['name'], 'color': get_color(category['name'])}
                     for category in ann_dict['categories']}
    direction_dict = {
        '39_1690.jpg': ['', '', '', 'bottom', 'right'],
        '2071.jpg': ['', 'bottom', '', 'bottom_right'],
        '3015.jpg': ['', 'bottom'],
    }
    for image in ann_dict['images']:
        if image['filename'] in direction_dict:
            direction_dict[image['id']] = [direction_dict[image['filename']], image]
    anns = defaultdict(list)
    for ann in ann_dict['annotations']:
        if ann['image_id'] in direction_dict:
            anns[ann['image_id']].append({
                'bbox': [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]],
                'name': category_dict[ann['category_id']]['name'],
                'color': category_dict[ann['category_id']]['color'],
            })
    for k in anns:
        anns[k].sort(key = lambda x: x['bbox'][0])
        for i in range(len(anns[k])):
            anns[k][i]['direction'] = direction_dict[k][0][i]
    for image_id in anns:
        draw_image(direction_dict[image_id][1], anns[image_id], images_path, image_output_path)


def main():
    data_dir = "/data/zhengwenhao/Datasets/object_detection/TCTDataSet"
    json_path = os.path.join(data_dir, 'coco/annotations/tct.json')
    images_path = os.path.join(data_dir, 'voc/JPEGImages')
    output_path = os.path.join(data_dir, 'middle_results/outputs')
    image_output_path = 'images'

    image_output_path = os.path.join(output_path, image_output_path)
    if os.path.exists(image_output_path):
        shutil.rmtree(image_output_path)
    os.makedirs(image_output_path)
    draw_images(json_path, images_path, image_output_path)


if __name__ == '__main__':
    temp_main()
