_base_ = [
    '../_base_/models/tct_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_hooks = []
model = dict(
    rpn_head = dict(abnormal = dict(
        loss_cls = dict(loss_weight = 0.0),
        loss_bbox = dict(loss_weight = 0.0))),
    roi_head = dict(abnormal = dict(bbox_head = dict(
        loss_cls = dict(loss_weight = 0.0),
        loss_bbox = dict(loss_weight = 0.0)))))


log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'tct_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'tct_rcnn', 'r50', '1x']))])
