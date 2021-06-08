_base_ = [
    '../_base_/models/tct_rcnn_r50_fpn.py',
    '../_base_/datasets/tct_detection_proposal.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
load_from = '/data/zhengwenhao/Result/TCT-RCNN/Model_result/TCT-RCNN/0.005lr/best_bbox_mAP_50_Normal_fcfirst_DifferenceFeature.pth'
custom_hooks = []

log_config = dict(
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'tct',
                                name = 'tct_rcnn_r50_fpn_1x_tct',
                                tags = ['mmdetection', 'tct', 'tct_rcnn', 'r50', '1x']))])
