_base_ = [
    '../_base_/models/faster_rcnn_vgg16_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(roi_head = dict(bbox_head = dict(num_classes = 5)))
optimizer = dict(lr = 0.005)

data = dict(
    samples_per_gpu = 3,
    workers_per_gpu = 3)

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'faster_rcnn_vgg16_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'faster_rcnn', 'vgg16', '1x']))])
