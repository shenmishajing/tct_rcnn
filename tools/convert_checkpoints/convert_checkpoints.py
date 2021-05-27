import re
import torch


def correct_ckpt(revise_keys = None):
    if revise_keys is None:
        revise_keys = [(r'^module\.', '')]
    revise_pattern = [(re.compile(k), v) for k, v in revise_keys]

    def wrapper(string):
        for p, v in revise_pattern:
            string = p.sub(v, string)
        return string

    return wrapper


def main():
    ckpt_path = '/data/zhengwenhao/Result/TCT-RCNN/Model_result/InceptionV3/best_acc.pth'
    ckpt_save_path = '/data/zhengwenhao/Result/TCT-RCNN/Model_result/YoloV3-Classifier/classifier.pth'
    ckpt = torch.load(ckpt_path)
    corrector = correct_ckpt([(r'^model\.', '')])
    ckpt['state_dict'] = {corrector(k): v for k, v in ckpt['state_dict'].items()}
    torch.save(ckpt, ckpt_save_path)


if __name__ == '__main__':
    main()
