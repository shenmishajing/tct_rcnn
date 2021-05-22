_base_ = [
    '../_base_/models/sparse_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(roi_head = dict(bbox_head = [dict(
    type = 'DIIHead',
    num_classes = 5,
    num_ffn_fcs = 2,
    num_heads = 8,
    num_cls_fcs = 1,
    num_reg_fcs = 3,
    feedforward_channels = 2048,
    in_channels = 256,
    dropout = 0.0,
    ffn_act_cfg = dict(type = 'ReLU', inplace = True),
    dynamic_conv_cfg = dict(
        type = 'DynamicConv',
        in_channels = 256,
        feat_channels = 64,
        out_channels = 256,
        input_feat_shape = 7,
        act_cfg = dict(type = 'ReLU', inplace = True),
        norm_cfg = dict(type = 'LN')),
    loss_bbox = dict(type = 'L1Loss', loss_weight = 5.0),
    loss_iou = dict(type = 'GIoULoss', loss_weight = 2.0),
    loss_cls = dict(
        type = 'FocalLoss',
        use_sigmoid = True,
        gamma = 2.0,
        alpha = 0.25,
        loss_weight = 2.0),
    bbox_coder = dict(
        type = 'DeltaXYWHBBoxCoder',
        clip_border = False,
        target_means = [0., 0., 0., 0.],
        target_stds = [0.5, 0.5, 1., 1.])) for _ in range(6)]))

# optimizer
optimizer = dict(_delete_ = True, type = 'AdamW', lr = 0.000025, weight_decay = 0.0001)
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 1, norm_type = 2))

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'sparse_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'sparse_rcnn', 'r50', '1x']))])
