_base_ = './tct_detection.py'
img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_rgb = True)
references = 3
ref_pipeline = [
    dict(type = 'LoadImageFromFile'),
    dict(type = 'LoadAnnotations', with_bbox = True),
    dict(type = 'Resize', img_scale = (1333, 800), keep_ratio = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'Normalize', **img_norm_cfg),
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'GroundTruthCrop'),
    dict(type = 'Resize', img_scale = (25, 25), keep_ratio = False, override = True),
    dict(type = 'DefaultFormatBundle'),
    dict(type = 'Collect', keys = ['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    train = dict(
        references = references,
        ref_pipeline = ref_pipeline),
    val = dict(
        references = references,
        ref_pipeline = ref_pipeline),
    test = dict(
        references = references,
        ref_pipeline = ref_pipeline))
