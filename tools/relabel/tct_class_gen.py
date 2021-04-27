import os
import json


def calculate_categories(part, categories):
    category_id = 0
    out_categories = []
    category_map = {}
    for category in categories:
        if part == '_normal' and category['name'] == 'NORMAL':
            category_map[category['id']] = category_id
            category['id'] = category_id
            category_id += 1
            out_categories.append(category)
        elif part == '_multi' and 'multi' in category['name']:
            category_map[category['id']] = category_id
            category['id'] = category_id
            category_id += 1
            out_categories.append(category)
        elif (part == '_single' or part == '') and category['name'] != 'NORMAL':
            if 'multi' not in category['name']:
                category_map[category['id']] = category_id
                category['id'] = category_id
                category_id += 1
                out_categories.append(category)
            elif part == '':
                category_map[category['id']] = category_map[category['id'] - len(out_categories)]
    return out_categories, category_map


def calculate_annotations(annotations, category_map):
    ann_id = 0
    out_annotations = []
    for ann in annotations:
        if ann['category_id'] in category_map:
            ann['category_id'] = category_map[ann['category_id']]
            ann['id'] = ann_id
            ann_id += 1
            out_annotations.append(ann)
    return out_annotations


def gen_tct_class_annotations(coco_path = 'coco'):
    for file in os.listdir(coco_path):
        if file.endswith('json'):
            name = file[:file.find('_')]
            for part in ['', '_single', '_multi', '_normal']:
                all_json = json.load(open(os.path.join(coco_path, file)))
                categories, category_map = calculate_categories(
                    part, all_json['categories'])
                all_json['categories'] = categories
                all_json['annotations'] = calculate_annotations(
                    all_json['annotations'], category_map)
                json.dump(all_json, open(os.path.join(
                    coco_path, name + part + '.json'), 'w'))


if __name__ == '__main__':
    data_dir = "/data/zhengwenhao/Datasets/TCTDataSet"
    coco_path = os.path.join(data_dir, "coco")
    gen_tct_class_annotations(coco_path)
