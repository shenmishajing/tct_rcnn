import os
import json
from collections import defaultdict
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont

import mmcv


def get_colors(num_colors):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(c * 256) for c in color))
    return colors


# 坐标顺序： 上-》左-》下-》右
def draw_bounding_box_on_image(draw,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               color = 'red',
                               thickness = 4,
                               display_str_list = ()):
    """
    Args:
      draw: a ImageDraw.Draw.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
    """
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    # 绘制Box框
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width = thickness, fill = color)

    # 加载字体
    try:
        font = ImageFont.truetype("font/simsun.ttc", 24, encoding = "utf-8")
    except IOError:
        font = ImageFont.load_default()

    # 计算显示文字的宽度集合 、高度集合
    display_str_width = [font.getsize(ds)[0] for ds in display_str_list]
    display_str_height = [font.getsize(ds)[1] for ds in display_str_list]
    # 计算显示文字的总宽度
    total_display_str_width = sum(display_str_width) + max(display_str_width) * 1.1
    # 计算显示文字的最大高度
    total_display_str_height = max(display_str_height)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # 计算文字背景框最右侧可到达的像素位置
    if right < (left + total_display_str_width):
        text_right = right
    else:
        text_right = left + total_display_str_width

    # 绘制文字背景框
    draw.rectangle(
        [(left, text_bottom), (text_right, text_bottom - total_display_str_height)],
        fill = color)

    # 计算文字背景框可容纳的文字，若超出部分不显示，改为补充“..”
    for index in range(len(display_str_list[::1])):
        current_right = (left + (max(display_str_width)) + sum(display_str_width[0:index + 1]))

        if current_right < text_right:
            display_str = display_str_list[:index + 1]
        else:
            display_str = display_str_list[0:index - 1] + '...'
            break

            # 绘制文字
    draw.text(
        (left + max(display_str_width) / 2, text_bottom - total_display_str_height),
        display_str,
        fill = 'black',
        font = font)


def draw_image(image, image_id_to_anns, images_path, image_output_path):
    if not len(image_id_to_anns[image['id']]):
        return
    img = Image.open(os.path.join(images_path, image['filename']))
    draw = ImageDraw.Draw(img)
    for bbox in image_id_to_anns[image['id']]:
        draw_bounding_box_on_image(draw, *bbox['bbox'], color = bbox['color'], display_str_list = bbox['name'])
    img.convert('RGB').save(os.path.join(image_output_path, image['filename']))


def draw_images(json_path, images_path, image_output_path):
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)

    ann_dict = json.load(open(json_path))
    colors = get_colors(len(ann_dict['categories']))
    category_dict = {category['id']: {'name': category['name'], 'color': color} for category, color in zip(ann_dict['categories'], colors)}
    image_id_to_anns = defaultdict(list)
    for ann in ann_dict['annotations']:
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
    json_path = os.path.join(data_dir, 'coco/tct_normal.json')
    images_path = os.path.join(data_dir, 'JPEGImages')
    output_path = os.path.join(data_dir, 'middle_results/outputs')
    image_output_path = 'images'

    image_output_path = os.path.join(output_path, image_output_path)
    draw_images(json_path, images_path, image_output_path)


if __name__ == '__main__':
    main()
