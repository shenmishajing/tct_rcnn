_base_ = [
    '../_base_/models/tct_rcnn_r50_fpn_normal.py',
    '../_base_/datasets/tct_detection_normal.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_hooks = []

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'tct_rcnn_r50_fpn_1x_tct_normal_head_only',
                                tags = ['mmdetection', 'tct', 'tct_rcnn', 'r50', '1x']))])
