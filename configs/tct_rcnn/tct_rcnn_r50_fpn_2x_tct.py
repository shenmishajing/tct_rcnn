_base_ = [
    '../_base_/models/tct_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

custom_hooks = []

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             init_kwargs = dict(project = 'tct',
                                name = 'tct_rcnn_r50_fpn_2x_tct',
                                tags = ['mmdetection', 'tct', 'tct_rcnn', 'r50', '2x']))])
