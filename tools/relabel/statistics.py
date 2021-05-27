import os
import json

from collections import defaultdict
from rich.console import Console
from rich.table import Table


def main():
    console = Console()
    round = 1
    data_root = '/home/zhengwenhao/Project/tct/mmdetection/data/tct/Normal_semi_supervision'
    for rate in [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 1]:
        print(rate)
        cur_data_root = os.path.join(data_root, 'round_{}/annotations_{:.2f}'.format(round, rate))
        json_path = os.path.join(cur_data_root, 'train_normal.json')
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


if __name__ == '__main__':
    main()
