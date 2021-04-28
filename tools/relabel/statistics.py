import os
import json

from collections import defaultdict
from rich.console import Console
from rich.table import Table


def static_numbers():
    console = Console()
    for part in ['tct', 'train', 'val', 'test']:
        print(part)
        json_path = f'coco/{part}_all.json'
        anns = json.load(open(json_path))
        category_dict = defaultdict(list)
        category_name_dict = {c['id']: c['name'] if 'multi' not in c['name'] else c['name'][:-6] for c in anns['categories']}
        for ann in anns['annotations']:
            category_dict[category_name_dict[ann['category_id']]].append(ann['image_id'])

        table = Table(show_header = True, header_style = "bold magenta")
        table.add_column("类别")
        table.add_column("框数量")
        table.add_column("图数量")
        for name in [c['name'] for c in anns['categories'] if 'multi' not in c['name']]:
            table.add_row(str(name), str(len(category_dict[name])), str(len(set(category_dict[name]))))

        console.print(table)


def main():
    data_dir = "/data/zhengwenhao/Datasets/TCTDataSet"
    json_path = os.path.join(data_dir, 'coco/tct_all.json')
    anns = json.load(open(json_path))
    category_dict = defaultdict(list)
    category_name_dict = {c['id']: c['name'] if 'multi' not in c['name'] else c['name'][:-6] for c in anns['categories']}
    for ann in anns['annotations']:
        category_dict[category_name_dict[ann['category_id']]].append(ann['image_id'])
    tct_images = set()
    for category in category_dict:
        if category == 'NORMAL':
            continue
        tct_images |= set(category_dict[category])
    all_images = set(img['id'] for img in anns['images'])
    normal_image = all_images - tct_images
    normal_annotations = {i: [] for i in normal_image}
    for ann in anns['annotations']:
        if ann['image_id'] in normal_annotations:
            normal_annotations[ann['image_id']].append(ann)
    print(normal_annotations)


if __name__ == '__main__':
    main()
