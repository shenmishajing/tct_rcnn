dataset_type = 'TCTDataset'
part = 'tct'
references = None
filter_min_size = 16
data_root = 'data/tct'
img_norm_cfg = dict(mean = [0, 0, 0], std = [255., 255., 255.], to_rgb = True)
train_pipeline = [
    dict(type = 'LoadImageFromFile', to_float32 = True),
    dict(type = 'LoadAnnotations', with_bbox = True),
    dict(type = 'PhotoMetricDistortion'),
    dict(
        type = 'Expand',
        mean = img_norm_cfg['mean'],
        to_rgb = img_norm_cfg['to_rgb'],
        ratio_range = (1, 2)),
    dict(
        type = 'MinIoURandomCrop',
        min_ious = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size = 0.3),
    dict(type = 'Resize', img_scale = [(320, 320), (608, 608)], keep_ratio = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'Normalize', **img_norm_cfg),
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'DefaultFormatBundle'),
    dict(type = 'Collect', keys = ['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type = 'LoadImageFromFile'),
    dict(
        type = 'MultiScaleFlipAug',
        img_scale = (608, 608),
        flip = False,
        transforms = [
            dict(type = 'Resize', keep_ratio = True),
            dict(type = 'RandomFlip'),
            dict(type = 'Normalize', **img_norm_cfg),
            dict(type = 'Pad', size_divisor = 32),
            dict(type = 'ImageToTensor', keys = ['img']),
            dict(type = 'Collect', keys = ['img'])
        ])
]
data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 4,
    train = dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = 'annotations',
        img_prefix = 'train2017',
        part = part,
        split = 'train',
        debug_len = None,
        filter_min_size = filter_min_size,
        references = references,
        pipeline = train_pipeline),
    val = dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = 'annotations',
        img_prefix = 'val2017',
        part = part,
        split = 'val',
        debug_len = None,
        filter_min_size = filter_min_size,
        references = references,
        pipeline = test_pipeline),
    test = dict(
        type = dataset_type,
        data_root = data_root,
        ann_file = 'annotations',
        img_prefix = 'val2017',
        part = part,
        split = 'test',
        debug_len = None,
        filter_min_size = filter_min_size,
        references = references,
        pipeline = test_pipeline))
evaluation = dict(interval = 1, metric = 'bbox', save_best = 'bbox_mAP_50', classwise = True)
