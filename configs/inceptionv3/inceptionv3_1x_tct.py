_base_ = [
    '../_base_/models/inceptionv3.py',
    '../_base_/datasets/tct_classification.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'inceptionv3_1x_tct',
                                tags = ['mmdetection', 'tct', 'inceptionv3', '1x']))])
