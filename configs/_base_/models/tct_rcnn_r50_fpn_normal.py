_base_ = 'tct_rcnn_r50_fpn.py'
model = dict(
    part = 'normal',
    roi_head = dict(
        normal = dict(test_cfg = dict(max_per_img = 100))))
