import xml.etree.ElementTree as ET
import os
import json

from tct_class_gen import gen_tct_class_annotations


def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids


def parseXmlFiles_by_txt(data_dir, json_save_path, split, categories, category_set):
    print(f'start parse split {split}')
    coco = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': categories
    }
    labelfile = split + ".txt"
    json_save_path = os.path.join(json_save_path, split + "_all.json")
    image_sets_file = data_dir + "/ImageSets/Main/" + labelfile
    ids = _read_image_ids(image_sets_file)

    image_set = set()

    image_id = 0
    annotation_id = 0
    for _id in ids:
        xml_file = data_dir + f"/Annotations/{_id}.xml"

        current_image_id = image_id
        current_category_id = None
        file_name = None

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception(
                'pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in image_set:
                    raise Exception('file_name duplicated')
            # add img item only after parse <size> tag
            elif elem.tag == 'size':
                if file_name is not None and elem[0].text is not None:
                    if file_name not in image_set:
                        size = {'width': int(
                            elem[0].text), 'height': int(elem[1].text)}
                        image_item = {'id': current_image_id,
                                      'filename': file_name, **size}
                        coco['images'].append(image_item)
                        image_set.add(file_name)
                        image_id += 1
                        print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception(
                            'duplicated image: {}'.format(file_name))
            elif elem.tag == 'object':
                # subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    if subelem.tag == 'name':
                        current_category_id = category_set[subelem.text]
                    elif subelem.tag == 'bndbox':
                        bndbox = {}
                        # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                        for option in subelem:
                            bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{}'.format(
                        current_image_id, current_category_id, bbox))
                    annotation_item = dict()
                    annotation_item['segmentation'] = []
                    seg = []
                    # bbox[] is x,y,w,h
                    # left_top
                    seg.append(bbox[0])
                    seg.append(bbox[1])
                    # left_bottom
                    seg.append(bbox[0])
                    seg.append(bbox[1] + bbox[3])
                    # right_bottom
                    seg.append(bbox[0] + bbox[2])
                    seg.append(bbox[1] + bbox[3])
                    # right_top
                    seg.append(bbox[0] + bbox[2])
                    seg.append(bbox[1])

                    annotation_item['segmentation'].append(seg)

                    annotation_item['area'] = bbox[2] * bbox[3]
                    annotation_item['iscrowd'] = 0
                    annotation_item['ignore'] = 0
                    annotation_item['image_id'] = current_image_id
                    annotation_item['bbox'] = bbox
                    annotation_item['category_id'] = current_category_id
                    annotation_item['id'] = annotation_id
                    coco['annotations'].append(annotation_item)
                    annotation_id += 1
    json.dump(coco, open(json_save_path, 'w'))


def gen_coco(voc_data_dir, json_save_path, cat_names_path):
    cat_names = [line.strip() for line in open(
        cat_names_path).readlines() if line.strip()]
    categories = []
    category_set = {}
    for i, c in enumerate(cat_names):
        category_item = {'supercategory': 'none', 'id': i, 'name': c}
        categories.append(category_item)
        category_set[c] = i
    for file in os.listdir(os.path.join(voc_data_dir, 'ImageSets/Main')):
        if file.endswith('.txt'):
            split = os.path.splitext(file)[0]
            parseXmlFiles_by_txt(voc_data_dir, json_save_path,
                                 split, categories, category_set)


def main():
    # 通过txt文件生成
    voc_data_dir = "/data/zhengwenhao/Datasets/TCTDataSet"
    json_save_path = os.path.join(voc_data_dir, "coco")
    cat_names_path = os.path.join(voc_data_dir, "obj.names")
    gen_coco(voc_data_dir, json_save_path, cat_names_path)
    gen_tct_class_annotations(json_save_path)


if __name__ == '__main__':
    main()
