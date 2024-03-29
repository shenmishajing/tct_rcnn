_base_ = './tct_detection.py'
# dataset
round = 1
# rate in [0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
rate = 0.1
ann_file = 'Normal_semi_supervision/round_{:d}/rate_{:.2f}'.format(round, rate)
part = 'normal'
data = dict(
    train = dict(
        ann_file = ann_file,
        part = part),
    val = dict(
        ann_file = ann_file,
        part = part),
    test = dict(
        split = 'test',
        ann_file = ann_file,
        part = part))
evaluation = dict(interval = 50)

# optimizer
optimizer = dict(type = 'SGD', lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
optimizer_config = dict(grad_clip = None)
# learning policy
lr_config = dict(
    policy = 'step',
    warmup = 'linear',
    warmup_iters = 500,
    warmup_ratio = 0.001,
    step = [800, 1100])
runner = dict(type = 'IterBasedRunner', max_iters = 1200)

checkpoint_config = dict(interval = 100)
# yapf:disable
log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type = 'NumClassCheckHook')]

dist_params = dict(backend = 'nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
