# The new config inherits a base config to highlight the necessary modification
#_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
_base_ = '/home/radu.berdan/work/kits/mmyolo-old/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'
#_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation

save_epoch_intervals = 1
train_batch_size_per_gpu = 16


pad_val = 0
img_scale = (512,512)
input_channels = 3

deepen_factor = 0.33
widen_factor = 0.25
last_stage_out_channels = 1024

img_type = 'rggb'

# simple1 - train on simple1 test on raw
data_root = '/home/radu.berdan/datasets/pascalraw_h400/'
train_data_prefix = 'raw/' if img_type == 'rggb' else 'pyraw_rgb/'
train_ann_file = 'annotations/flip_dng_instances_train.json'
val_data_prefix = 'raw/' if img_type == 'rggb' else 'pyraw_rgb/'
val_ann_file = 'annotations/flip_dng_instances_test.json'

data_root = '/home/radu.berdan/datasets/cocoraw-v1/'
train_data_prefix = 'images/train2017/' if img_type == 'rggb' else 'pyraw_rgb/'
train_ann_file = 'annotations/instances_train2017_3cat.json'
val_data_prefix = 'images/val2017/' if img_type == 'rggb' else 'pyraw_rgb/'
val_ann_file = 'annotations/instances_val2017_3cat.json'

model = dict(
    data_preprocessor=dict(
        mean=[],
        std=[],
        bgr_to_rgb=False),
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        input_channels=input_channels),
        #input_channels=3),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],
            num_classes=80)),
    train_cfg=dict(
        assigner=dict(
            num_classes=80)))


file_client_args = dict(backend='disk')

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args, img_type=img_type),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline=[
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=pad_val)),
    *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file,
        pipeline=train_pipeline))

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')

test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args, img_type=img_type),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=pad_val)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline))

train_cfg = dict(
    max_epochs=1,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[])

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals))

optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

# param_scheduler = [dict(
#     type='MultiStepLR',
#     by_epoch=True,  # Updated by epochs

#     milestones=[48],
#     gamma=0.1)]

# default_hooks = dict(
#     logger=dict(
#         type='LoggerHook',
#         interval=1,
#         out_dir='./runs/hello',
#         log_metric_by_epoch=True
#     ))

# default_hooks = dict(
#     logger=dict(
#         type='TensorboardLoggerHook'))

optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=8)