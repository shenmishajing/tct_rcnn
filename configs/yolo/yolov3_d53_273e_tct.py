_base_ = [
    '../_base_/models/yolov3_d53.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_273e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(bbox_head = dict(num_classes = 5))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'yolov3_d53_273e_tct',
                                tags = ['mmdetection', 'tct', 'yolov3', 'd53', '273e']))])
