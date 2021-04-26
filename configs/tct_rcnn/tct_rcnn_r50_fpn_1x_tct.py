_base_ = [
    '../_base_/models/tct_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_hooks = []
model = dict(roi_head = dict(bbox_head = dict(tct = dict(num_relation_parts = 16))))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'tct_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'tct_rcnn', 'r50', '1x']))])
