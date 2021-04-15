_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
runner = dict(type = 'EpochBasedRunner', max_epochs = 36)
model = dict(rpn_head = dict(loss_bbox = dict(type = 'L1Loss', loss_weight = 1.0)),
             roi_head = dict(bbox_head = dict(num_classes = 5)))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'faster_rcnn_r50_fpn_2x_tct',
                                tags = ['mmdetection', 'tct', 'faster_rcnn', 'r50', '2x']))])
