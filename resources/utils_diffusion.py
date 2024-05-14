import json
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


model_paths_rggb = {
    "n": "/home/radu.berdan/work/models/rggb/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb-rggb.pth",
    "s": "/home/radu.berdan/work/models/rggb/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1-rggb.pth",
    "m": "/home/radu.berdan/work/models/rggb/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a-rggb.pth",
    "l": "/home/radu.berdan/work/models/rggb/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6-rggb.pth",
    "x": "/home/radu.berdan/work/models/rggb/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c-rggb.pth",
}

model_paths = {
    "n": "/home/radu.berdan/work/models/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth",
    "s": "/home/radu.berdan/work/models/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth",
    "m": "/home/radu.berdan/work/models/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth",
    "l": "/home/radu.berdan/work/models/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth",
    "x": "/home/radu.berdan/work/models/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth",
}

factors = {
    "n": {"d": 0.33, "w": 0.25},
    "s": {"d": 0.33, "w": 0.5},
    "m": {"d": 0.67, "w": 0.75},
    "l": {"d": 1, "w": 1},
    "x": {"d": 1, "w": 1.25},
}

start_epoch = {"n": 497, "s": 430, "m": 460, "l": 460, "x": 450}

last_stage_channels = {"n": 1024, "s": 1024, "m": 768, "l": 512, "x": 512}

mean_cocoraw_pascal = [214.914, 486.741, 487.646, 439.222]
std_cocoraw_pascal = [23.559, 98.175, 99.328, 86.943]

mean_cocoraw = [1318.185, 2695.745, 2703.114, 2168.397]
std_cocoraw = [163.359, 589.635, 592.687, 450.485]


def update_cfg(cfg, params):

    cfg.model["data_preprocessor"]["type"] = 'YOLOv5DetDataPreprocessor_RAWtoRGB'
    cfg.model["data_preprocessor"]["mean"] = params['mean']
    cfg.model["data_preprocessor"]["std"] = params['std']
    cfg.model["data_preprocessor"]["bgr_to_rgb"] = params['bgr_to_rgb']
    cfg.model["backbone"]["input_channels"] = params['input_channels']

    batch_size = params["batch_size"] // params["grad_div"]

    cfg["optim_wrapper"]["optimizer"]["lr"] = params["lr"]

    deepen_factor = factors[params["yolo_type"]]["d"]
    widen_factor = factors[params["yolo_type"]]["w"]
    last_stage_out_channels = last_stage_channels[params["yolo_type"]]

    cfg.model["backbone"]["type"] = "YOLOv8CSPDarknet"
    cfg.model["backbone"]["last_stage_out_channels"] = last_stage_out_channels
    cfg.model["neck"]["in_channels"] = [256, 512, last_stage_out_channels]
    cfg.model["neck"]["out_channels"] = [256, 512, last_stage_out_channels]
    cfg.model["bbox_head"]["head_module"]["in_channels"] = [
        256,
        512,
        last_stage_out_channels,
    ]

    cfg.model["backbone"]["deepen_factor"] = deepen_factor
    cfg.model["backbone"]["widen_factor"] = widen_factor
    cfg.model["backbone"]["preproc"] = params["preproc"]
    cfg.model["backbone"]["preproc_params"] = params["preproc_params"]

    cfg.model["neck"]["deepen_factor"] = deepen_factor
    cfg.model["neck"]["widen_factor"] = widen_factor

    cfg.model["bbox_head"]["head_module"]["widen_factor"] = widen_factor
    cfg.model["bbox_head"]["head_module"]["num_classes"] = params["num_classes"]
    cfg.model["train_cfg"]["assigner"]["num_classes"] = params["num_classes"]

    cfg.train_dataloader["batch_size"] = batch_size
    cfg.train_cfg["max_epochs"] = params["max_epochs"]
    cfg.optim_wrapper["optimizer"]["lr"] = params["lr"]
    cfg.optim_wrapper["optimizer"]["batch_size_per_gpu"] = batch_size

    cfg.default_hooks["logger"]["out_dir"] = params["run_path"]
    cfg.optimizer_config["cumulative_iters"] = params["grad_div"]

    # ########################################################

    img_scale = params["img_scale"]
    preproc = params["preproc"]
    pad_val = params["pad_val"]

    file_client_args = dict(backend="disk")

    load_image_type = "LoadNumpyFromFile"

    pre_transform = [
        dict(
            type=load_image_type,
        ),

        dict(type="LoadAnnotations", with_bbox=True),
    ]


    last_transform = [
        dict(
            type="mmdet.PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "flip",
                "flip_direction",
            ),
        ),
    ]

    img_scale = params['img_scale']
    affine_scale = params['affine_scale']

    train_pipeline = [
        *pre_transform,
        dict(type="YOLOv5KeepRatioResize", scale=img_scale),
        dict(
            type="LetterResize",
            scale=img_scale,
            allow_scale_up=False,
            pad_val=dict(img=pad_val),
        ),
        # dict(
        #     type="Mosaic",
        #     img_scale=img_scale,
        #     pad_val=params['pad_val'],
        #     pre_transform=pre_transform),
        
        # dict(
        #     type='YOLOv5RandomAffine',
        #     max_rotate_degree=0.0,
        #     max_shear_degree=0.0,
        #     scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        #     max_aspect_ratio=params['max_aspect_ratio'],
        #     # img_scale is (width, height)
        #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
        #     border_val=(params['pad_val'], params['pad_val'], params['pad_val'])),
        # dict(type="YOLOv5KeepRatioResize", scale=img_scale),
        # dict(
        #     type="LetterResize",
        #     scale=img_scale,
        #     allow_scale_up=False,
        #     pad_val=dict(img=pad_val),
        # ),
        dict(type='RandomFlip', prob=0.5),
        *last_transform,
    ]

    test_pipeline = [
        *pre_transform,
        dict(
            type="LetterResize",
            scale=img_scale,
            allow_scale_up=False,
            pad_val=dict(img=pad_val),
        ),
        dict(
            type="mmdet.PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "pad_param",
            ),
        ),
    ]

    dataset=dict(
        type="SampleConcatDataset",
        n=3206,
        f=params["f"],
        datasets=[
            dict(
                type=params["dataset_type"],
                data_root=params["nod_data_root_train"],
                # ann_file='annotations/raw_new_Nikon750_train.json',
                ann_file=params["nod_train_ann_file"],
                #data_prefix=dict(img='raw_new_Nikon750_train/'),
                data_prefix=params["train_data_prefix"],
                pipeline=train_pipeline,
            ),
            dict(
                type=params["dataset_type"],
                data_root=params["city_data_root_train"],
                dataset_name=params["city_dataset_name"],
                ann_file=params["city_train_ann_file"],
                data_prefix=params['data_prefix'], ###################
                #data_prefix=dict(img='images/100k/train/'),
                pipeline=train_pipeline,
                # normalize_reverse=(255, 16383),
                # replace_filename={".png": "_pred.npy"},
                replace_filename=params['replace_filename'], ###################
                #replace_filename={".png": ".npy"}, # FAL
                #normalization_align_type="basic"
                normalization_align_type=params['norm_align'],
                upsample_factor=params['upsample_factor']
            )
        ]
    )


    # val_dataloader = dict(
    #     batch_size=1,
    #     num_workers=6,
    #     persistent_workers=True,
    #     drop_last=False,
    #     sampler=dict(type='DefaultSampler', shuffle=False),
    #     dataset=dict(
    #         type=params["dataset_type"],
    #         data_root=params["data_root_test"] ,
    #         ann_file='annotations/raw_new_Nikon750_test.json',
    #         data_prefix=dict(img='raw_new_Nikon750_test/'),
    #         test_mode=True,
    #         pipeline=test_pipeline))
    # test_dataloader = dict(
    #     batch_size=1,
    #     num_workers=6,
    #     persistent_workers=True,
    #     drop_last=False,
    #     sampler=dict(type='DefaultSampler', shuffle=False),
    #     dataset=dict(
    #         type=params["dataset_type"],
    #         data_root=params["data_root_test"],
    #         ann_file='annotations/raw_new_Nikon750_test.json',
    #         data_prefix=dict(img='raw_new_Nikon750_test/'),
    #         test_mode=True,
    #         pipeline=test_pipeline))


    cfg.train_dataloader["dataset"] = dataset
    # cfg.test_dataloader = test_dataloader
    # cfg.val_dataloader = val_dataloader

    cfg.val_dataloader["dataset"]["pipeline"] = test_pipeline
    cfg.val_dataloader["dataset"]["data_root"] = params["data_root_test"]
    cfg.val_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    cfg.val_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    cfg.test_dataloader["dataset"]["pipeline"] = test_pipeline
    cfg.test_dataloader["dataset"]["data_root"] = params["data_root_test"]
    cfg.test_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    cfg.test_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    cfg.val_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]
    cfg.test_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]


    # cfg.val_dataloader["dataset"]["pipeline"] = test_pipeline
    # cfg.val_dataloader["dataset"]["data_root"] = params["data_root_test"]
    # cfg.val_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    # cfg.val_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    # cfg.test_dataloader["dataset"]["pipeline"] = test_pipeline
    # cfg.test_dataloader["dataset"]["data_root"] = params["data_root_test"]
    # cfg.test_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    # cfg.test_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    # cfg.val_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]
    # cfg.test_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]

    return cfg

def update_cfg_augment(cfg, params):
    albu_train_transforms = [
        dict(type='Blur', p=0.01),
        dict(type='MedianBlur', p=0.01),
        #dict(type='ToGray', p=0.01),
        #dict(type='CLAHE', p=0.01)
    ]

    last_transform = [
        # dict(
        #     type='mmdet.Albu',
        #     transforms=albu_train_transforms,
        #     bbox_params=dict(
        #         type='BboxParams',
        #         format='pascal_voc',
        #         label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        #     keymap={
        #         'img': 'image',
        #         'gt_bboxes': 'bboxes'
        #     }),
        #dict(type='YOLOv5HSVRandomAug'),
        
        dict(type='mmdet.RandomFlip', prob=0.5),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
    ]

    if "rggb-mix" == params["model_type"]:
        load_image_type = "LoadImageFromFileRAWmix"
    else:
        load_image_type = (
            "LoadImageFromFileRAW"
            if params["model_type"] == "rggb"
            else "LoadImageFromFileRGB"
        )
        mosaic_type = 'MosaicRAW' if params["model_type"] == "rggb" else 'Mosaic'
    file_client_args = dict(backend="disk")
    pre_transform = [
        dict(
            type=load_image_type,
            file_client_args=file_client_args,
            scale=1,
        ),
        dict(type="LoadAnnotations", with_bbox=True),
    ]

    img_scale = params['img_scale']
    affine_scale = params['affine_scale']

    train_pipeline = [
        *pre_transform,
        dict(
            type='Mosaic',
            img_scale=img_scale,
            pad_val=params['pad_val'],
            pre_transform=pre_transform),
        # dict(
        #     type='YOLOv5RandomAffine',
        #     max_rotate_degree=0.0,
        #     max_shear_degree=0.0,
        #     scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        #     max_aspect_ratio=params['max_aspect_ratio'],
        #     # img_scale is (width, height)
        #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
        #     border_val=(params['pad_val'], params['pad_val'], params['pad_val'])),
        *last_transform
    ]

    cfg.train_dataloader["dataset"]["pipeline"] = train_pipeline

    return cfg




def make_name(params):
    """This function does something"""
    yolo_type = params["yolo_type"]
    pretrained = "1" if params["pretrained"] else "0"
    lr = params["lr"]
    epochs = params["max_epochs"]
    batch_size = params["batch_size"]
    grad_div = params["grad_div"]
    dataset_type = params["dataset_type"]
    model_type = params["model_type"]
    preproc = params["preproc"]

    name = f"yolov8_{yolo_type}_e-{epochs}_lr-{lr}_b-{batch_size}_p-{pretrained}_data-{dataset_type}_stem-{model_type}"

    if params["backbone"] == "res":
        name += "_resnet"
    return name


def convert_to_tboard(project_path, name, tboard_dir):
    print(project_path)

    for directory in os.listdir(project_path):
        if "2023" in directory and "." not in directory:
            file_name = directory
            break

    file_path = f"{project_path}/{file_name}/vis_data/{file_name}.json"

    metrics = {
        "coco/bbox_mAP": [],
        "coco/bbox_mAP_50": [],
        "coco/bbox_mAP_75": [],
        "coco/bbox_mAP_s": [],
        "coco/bbox_mAP_m": [],
        "coco/bbox_mAP_l": [],
    }

    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            decoded = json.loads(line)
            if "coco/bbox_mAP" in decoded.keys():
                for key in metrics.keys():
                    metrics[key].append(decoded[key])

    n_epochs = len(metrics["coco/bbox_mAP"])

    run_path = f"./{tboard_dir}/{name}"

    writer = SummaryWriter(run_path)

    for i in range(n_epochs):
        for key in metrics.keys():
            writer.add_scalar(key, metrics[key][i], i)


def cosine(epoch, stop, lr_max, lr_min):
    if epoch >= stop:
        lr = lr_min
    else:
        lr = (1 + np.cos(np.pi * epoch / (stop - 1))) / 2 * (lr_max - lr_min) + lr_min
    return lr


def cosine_var(epoch, start, stop, lr_max, lr_min):
    if epoch < start:
        lr = lr_max
    elif epoch >= stop:
        lr = lr_min
    else:
        e = epoch - start
        end = stop - start
        lr = (1 + np.cos(np.pi * e / (end - 1))) / 2 * (lr_max - lr_min) + lr_min
    return lr


def linear_lr(epoch, start=0, stop=500, lr_max=0.01, lr_min=0.001):
    if epoch < start:
        lr = lr_max
    elif epoch >= stop:
        lr = lr_min
    else:
        lr = lr_max - epoch / stop * (lr_max - lr_min)
    return lr


def update_testset(cfg, params):
    cfg.val_dataloader["dataset"]["data_root"] = params["data_root_test"]
    cfg.val_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    cfg.val_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    cfg.test_dataloader["dataset"]["data_root"] = params["data_root_test"]
    cfg.test_dataloader["dataset"]["data_prefix"]["img"] = params["test_data_prefix"]
    cfg.test_dataloader["dataset"]["ann_file"] = params["test_ann_file"]

    cfg.val_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]
    cfg.test_evaluator["ann_file"] = params["data_root_test"] + params["test_ann_file"]

    return cfg

def fun(x, gamma):
    return x**gamma

def make_cnn_gamma(gamma, channels, in_chans=4):
    # Choose gamma value and number of channels in the conv layer

    # Compute weight and bias weight (d) values
    bias = np.linspace(0, 1, channels+1)
    bias = bias ** 2 # since gamma is 'bendy' close to 0, have more bias values close to 0.

    weights = np.zeros(channels)

    for i in range(len(weights)):
        w = (fun(bias[i+1], gamma)-fun(bias[i], gamma)) / (bias[i+1]-bias[i]) - np.sum(weights)
        weights[i] = w

    #weights = np.concatenate([weights]*in_chans)
    weights = np.tile(weights, (in_chans, 1))
    weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1).astype(np.float32)
    bias = np.concatenate([bias[:-1]]*in_chans).astype(np.float32)

    print('->>>', weights)
    print('->>>', bias)

    return weights, bias
