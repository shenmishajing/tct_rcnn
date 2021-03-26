_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(roi_head = dict(bbox_head = [
    dict(
        type = 'Shared2FCBBoxHead',
        in_channels = 256,
        fc_out_channels = 1024,
        roi_feat_size = 7,
        num_classes = 5,
        bbox_coder = dict(
            type = 'DeltaXYWHBBoxCoder',
            target_means = [0., 0., 0., 0.],
            target_stds = [0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic = True,
        loss_cls = dict(
            type = 'CrossEntropyLoss',
            use_sigmoid = False,
            loss_weight = 1.0),
        loss_bbox = dict(type = 'SmoothL1Loss', beta = 1.0,
                         loss_weight = 1.0)),
    dict(
        type = 'Shared2FCBBoxHead',
        in_channels = 256,
        fc_out_channels = 1024,
        roi_feat_size = 7,
        num_classes = 5,
        bbox_coder = dict(
            type = 'DeltaXYWHBBoxCoder',
            target_means = [0., 0., 0., 0.],
            target_stds = [0.05, 0.05, 0.1, 0.1]),
        reg_class_agnostic = True,
        loss_cls = dict(
            type = 'CrossEntropyLoss',
            use_sigmoid = False,
            loss_weight = 1.0),
        loss_bbox = dict(type = 'SmoothL1Loss', beta = 1.0,
                         loss_weight = 1.0)),
    dict(
        type = 'Shared2FCBBoxHead',
        in_channels = 256,
        fc_out_channels = 1024,
        roi_feat_size = 7,
        num_classes = 5,
        bbox_coder = dict(
            type = 'DeltaXYWHBBoxCoder',
            target_means = [0., 0., 0., 0.],
            target_stds = [0.033, 0.033, 0.067, 0.067]),
        reg_class_agnostic = True,
        loss_cls = dict(
            type = 'CrossEntropyLoss',
            use_sigmoid = False,
            loss_weight = 1.0),
        loss_bbox = dict(type = 'SmoothL1Loss', beta = 1.0, loss_weight = 1.0))
]))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             init_kwargs = dict(project = 'tct',
                                name = 'cascade_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'cascade_rcnn', 'r50', '1x']))])
