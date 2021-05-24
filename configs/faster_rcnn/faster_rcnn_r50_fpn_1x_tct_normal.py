_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection_normal.py'
]
model = dict(roi_head = dict(bbox_head = dict(num_classes = 1)))
optimizer = dict(lr = 0.005)

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'faster_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'faster_rcnn', 'r50', '1x']))])
