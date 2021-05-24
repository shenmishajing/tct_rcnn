import os
import math
import json
import shutil
import random


def gen_rate(data_root, folder_name_tmp, rate):
    src_path = os.path.join(data_root, folder_name_tmp.format(1))
    des_path = os.path.join(data_root, folder_name_tmp.format(rate))

    if os.path.exists(des_path):
        return
    os.makedirs(des_path)

    for file in os.listdir(src_path):
        if file != 'train_normal.json':
            shutil.copy2(os.path.join(src_path, file), os.path.join(des_path, file))
            continue
        ann = json.load(open(os.path.join(src_path, file)))
        random.shuffle(ann['annotations'])
        ann['annotations'] = sorted(ann['annotations'][:int(math.ceil(len(ann['annotations']) * rate))], key = lambda x: x['id'])
        for i in range(len(ann['annotations'])):
            ann['annotations'][i]['id'] = i
        json.dump(ann, open(os.path.join(des_path, file), 'w'))


def main():
    data_root = '/home/zhengwenhao/Project/tct/mmdetection/data/tct/Normal_semi_supervision/round_1'
    rate = [0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 0.6, 0.75]
    folder_name_tmp = 'annotations_{:.2f}'
    for r in rate:
        gen_rate(data_root, folder_name_tmp, r)


if __name__ == '__main__':
    main()
