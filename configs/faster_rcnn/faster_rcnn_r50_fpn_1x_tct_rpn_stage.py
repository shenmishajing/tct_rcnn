_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(rpn_head = dict(loss_cls = dict(type = 'CrossEntropyLoss', use_sigmoid = True, loss_weight = 1.0),
                             loss_bbox = dict(type = 'L1Loss', loss_weight = 30.0)),
             roi_head = dict(bbox_head = dict(num_classes = 5,
                                              loss_cls = dict(type = 'CrossEntropyLoss', use_sigmoid = False, loss_weight = 0.0),
                                              loss_bbox = dict(type = 'L1Loss', loss_weight = 0.0))))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'faster_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'faster_rcnn', 'r50', '1x']))])
