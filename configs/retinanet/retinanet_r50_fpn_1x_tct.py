_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type = 'SGD', lr = 0.005, momentum = 0.9, weight_decay = 0.0001)

model = dict(bbox_head = dict(num_classes = 5))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'retinanet_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'retinanet', 'r50', '1x']))])
