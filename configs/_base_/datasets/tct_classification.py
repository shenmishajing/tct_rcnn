dataset_type = 'TCTClassificationDataset'
filter_min_size = 16
data_root = 'data/tct/'
img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_rgb = True)
train_pipeline = [
    dict(type = 'LoadImageFromFile'),
    dict(type = 'LoadAnnotations', with_bbox = True),
    dict(type = 'Resize', img_scale = (1333, 800), keep_ratio = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'Normalize', **img_norm_cfg),
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'GroundTruthCrop'),
    dict(type = 'Resize', img_scale = (299, 299), keep_ratio = False, override = True),
    dict(type = 'DefaultFormatBundle'),
    dict(type = 'Collect', keys = ['img', 'gt_labels']),
]
test_pipeline = [
    dict(type = 'LoadImageFromFile'),
    dict(
        type = 'MultiScaleFlipAug',
        img_scale = (1333, 800),
        flip = False,
        transforms = [
            dict(type = 'Resize', keep_ratio = True),
            dict(type = 'RandomFlip'),
            dict(type = 'Normalize', **img_norm_cfg),
            dict(type = 'Pad', size_divisor = 32),
            dict(type = 'GroundTruthCrop'),
            dict(type = 'Resize', img_scale = (299, 299), keep_ratio = False, override = True),
            dict(type = 'ImageToTensor', keys = ['img']),
            dict(type = 'Collect', keys = ['img']),
        ])
]
data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 4,
    train = dict(
        type = dataset_type,
        ann_file = data_root + 'annotations/train.json',
        img_prefix = data_root + 'train2017/',
        filter_min_size = filter_min_size,
        pipeline = train_pipeline),
    val = dict(
        type = dataset_type,
        ann_file = data_root + 'annotations/val.json',
        img_prefix = data_root + 'val2017/',
        filter_min_size = filter_min_size,
        pipeline = test_pipeline),
    test = dict(
        type = dataset_type,
        ann_file = data_root + 'annotations/test.json',
        img_prefix = data_root + 'val2017/',
        filter_min_size = filter_min_size,
        pipeline = test_pipeline))
evaluation = dict(interval = 1, rule = 'greater', save_best = 'acc')
