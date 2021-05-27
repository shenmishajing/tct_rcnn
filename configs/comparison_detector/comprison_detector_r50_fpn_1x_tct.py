_base_ = [
    '../_base_/models/comparison_detector_r50_fpn.py',
    '../_base_/datasets/tct_detection_reference.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(roi_head = dict(bbox_head = dict(num_classes = 5)))
optimizer = dict(lr = 0.005)

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'comparison_detector_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'comparison_detector', 'r50', '1x']))])
