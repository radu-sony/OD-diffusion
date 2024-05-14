import json

file_names = [
    {"preproc_type": "a", "dataset": "nod", "file_name": "fundiv_nod.json"},
    {"preproc_type": "a", "dataset": "pascal", "file_name": "fundiv_pascal.json"},
    {"preproc_type": "g", "dataset": "nod", "file_name": "gamma_nod.json"},
    {"preproc_type": "g", "dataset": "pascal", "file_name": "gamma_pascal.json"},
]

norms_preproc = {"nod": {}, "pascal": {}}
for norms_file in file_names:
    file_name = norms_file["file_name"]
    with open(
        f"/home/radu.berdan/work/OD/configs/normalization/{file_name}", "r"
    ) as f:
        data = json.load(f)
    norms_preproc[norms_file["dataset"]][norms_file["preproc_type"]] = data


file_names_cocoraw = [
    {"preproc_type": "a", "dataset": "nod", "file_name": "fundiv_cocoraw-nod.json"},
    {
        "preproc_type": "a",
        "dataset": "pascal",
        "file_name": "fundiv_cocoraw-pascal.json",
    },
    {"preproc_type": "g", "dataset": "nod", "file_name": "gamma_cocoraw-nod.json"},
    {
        "preproc_type": "g",
        "dataset": "pascal",
        "file_name": "gamma_cocoraw-pascal.json",
    },
]

norms_preproc_cocoraw = {"nod": {}, "pascal": {}}
for norms_file in file_names_cocoraw:
    file_name = norms_file["file_name"]
    with open(
        f"/home/radu.berdan/work/OD/configs/normalization/{file_name}", "r"
    ) as f:
        data = json.load(f)
    norms_preproc_cocoraw[norms_file["dataset"]][norms_file["preproc_type"]] = data


params = {
    "yolo_type": "n",
    "pretrained": False,
    "lr": 0.001,
    "max_epochs": 16,
    "num_classes": 3,  # person, car, bicycle
    "milestones": [60],
    "lr_scale": 0.1,
    "run_path": "",
    "work_path": "",
    "batch_size": 64,
    "grad_div": 2,
    "model_type": "rggb",  # raw, rggb
    "backbone": "dark",
    "dataset_type": "rggb",
    "img_scale": (512, 512),
    "pad_val": 0,
    "rggb_max_train": 2**14,
    "rggb_max_test": 2**14,
    "preproc": "dynamic-gamma",
    "preproc_params": {},
    "data_root_train": "",
    "train_data_prefix": "",
    "train_ann_file": "",
    "data_root_test": "",
    "test_data_prefix": "",
    "test_ann_file": "",
    "affine_scale": 0.5,
    "max_aspect_ratio": 100
}

norm_link = {
    "cocoraw-v1+pascal": "cocoraw-v1",
    "cocoraw-v1-pascal+pascal": "cocoraw-v1-pascal",
    "cocoraw-v1-nod+pascal": "cocoraw-v1-nod",
    "cocoraw-v1+nod": "cocoraw-v1",
    "cocoraw-v1-nod+nod": "cocoraw-v1-nod",
    "cocoraw-v1": "cocoraw-v1",
    "rod-day": "rod",
    "rod-night": "rod"
}

norms = {
    "cocoraw-v1": {
        "rggb_max": 2**14,
        "mean": [0.07999858, 0.16160948, 0.16202903, 0.12901926],
        "std": [0.00989179, 0.03576194, 0.03592445, 0.02728775],
    },
    "cocoraw-v1-nod": {
        "rggb_max": 2**14,
        "mean": [0.10164573, 0.18464398, 0.18523392, 0.14850278],
        "std": [0.01161111, 0.0380951, 0.03821255, 0.02815485],
    },
    "cocoraw-v1-pascal": {
        "rggb_max": 2**12,
        "mean": [0.05423013, 0.1232285, 0.12346591, 0.11167766],
        "std": [0.00626083, 0.02591333, 0.02623225, 0.02317211],
    },
    "pascalraw": {
        "rggb_max": 2**12,
        "mean": [0.05076418, 0.10138614, 0.10139126, 0.08010517],
        "std": [0.00449451, 0.0142749, 0.01428685, 0.01051754],
    },
    "nod": {
        "rggb_max": 2**14,
        "mean": [0.04933929, 0.05516977, 0.05525395, 0.04697527],
        "std": [0.00230979, 0.00341231, 0.00340906, 0.0020846],
    },
    "rod": {
        "rggb_max": 1.,
        "mean": [0.04933929, 0.05516977, 0.05525395, 0.04697527],
        "std": [0.00230979, 0.00341231, 0.00340906, 0.0020846],
    },
}

paths = {
    "pascal": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "train_ann_file": "annotations/flip_dng_instances_test_tight.json",
        "data_root_test": "/home/radu.berdan/datasets/pascalraw_h400a/",
        "test_data_prefix": "raw/",
        "test_ann_file": "annotations/flip_dng_instances_test_tight.json",
    },
    "nod": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "data_root_test": "/home/radu.berdan/datasets/NOD_h400/",
        "test_data_prefix": "raw/",
        "test_ann_file": "annotations/raw_new_Nikon750_test_flip.json",
    },
    "cocoraw-v1+pascal+nod": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "data_root_test": "/home/radu.berdan/datasets/NOD_h400/",
        "test_data_prefix": "raw/",
        "test_ann_file": "annotations/raw_new_Nikon750_test_flip.json",
    },
    "cocoraw-v1": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "data_root_test": "/home/radu.berdan/datasets/",
        "test_data_prefix": "",
        "test_ann_file": "annotations/cocoraw-v1_val.json",
    },
    "rod-day": {
        "data_root_train": "/home/data/ROD_dataset/",
        "train_data_prefix": "images/",
        "train_ann_file": "annotations/day-train-anno.json",
        "data_root_test": "/home/data/ROD_dataset/",
        "test_data_prefix": "images/",
        "test_ann_file": "annotations/day-val-anno.json",
    },
    "rod-night": {
        "data_root_train": "/home/data/ROD_dataset/",
        "train_data_prefix": "images/",
        "train_ann_file": "annotations/night-train-anno.json",
        "data_root_test": "/home/data/ROD_dataset/",
        "test_data_prefix": "images/",
        "test_ann_file": "annotations/night-val-anno.json",
    },
}

norms_rgb = {
    "coco": {
        "rgb_max": 2**8,
        "mean": [0.46845597, 0.4461054, 0.40788974],
        "std": [0.07716215, 0.07445913, 0.08381168],
    }
}

paths_rgb = {
    "pascal": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "data_root_test": "/home/radu.berdan/datasets/pascalraw_h400/",
        "test_data_prefix": "rgb/",
        "test_ann_file": "annotations/flip_dng_instances_test_tight.json",
    },
    "nod": {
        "data_root_train": "/home/radu.berdan/datasets/",
        "train_data_prefix": "",
        "data_root_test": "/home/radu.berdan/datasets/NOD_h400/",
        "test_data_prefix": "rgb/",
        "test_ann_file": "annotations/raw_new_Nikon750_test_flip.json",
    },
}
