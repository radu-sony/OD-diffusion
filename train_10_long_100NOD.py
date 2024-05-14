import torch
import sys
cuda_no = str(sys.argv[2])

device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)

import time

from mmengine.config import Config
from mmyolo.registry import RUNNERS

#from resources.utils_diffusion import update_cfg, linear_lr
from resources.utils import update_cfg, linear_lr

from configs.params_init import params, norms, paths, norm_link
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from tqdm import tqdm
import numpy as np
import resources.models
import resources.transforms
import resources.dataset_wrappers

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

params["lr"] = 0.01
params["lr_min"] = 0.001
params["max_epochs_scaling"] = 50
params["max_epochs"] = 40
params["start_cosine_epochs"] = 28
params["stop_cosine_epochs"] = 36
params["img_scale"] = (640, 416)

params['preproc'] = None
params['dataset_type'] = 'rggb'


params["preproc_params"] = {
    "name": None,
    "value": 1,
    "rggb_max": 2**14,
    "hidden_size": 32,
    "n_layers": 4,
    "n_params": 3,
    "mean": [],
    "std": [],
    "scale_test": 1,
}

params["num_classes"] = 3


CFG_PATH = "./configs/yolov8_n_pascalraw.py"
TBOARD_DIR = "./runs/runs-3-diffusion-test/"

DISABLE = False

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 64

base_datasets = ["nod"]
preproc_fun = 'norm-max'

names = {'Sony': 'Sony_RX100m7',
        'Nikon': 'Nikon750'}

camera_names = ['Nikon']
dataset_name = "cocoraw-v1-nod+nod"

n_runs = 3

for run_no in range(n_runs):
    for camera_name in camera_names:

        camera_name_full = names[camera_name]
        data_root_train = "/home/radu.berdan/datasets/NOD_h416_d32/"
        train_data_prefix = f'raw_new_{camera_name_full}_train/'
        train_ann_file = f'annotations/raw_new_{camera_name_full}_train_100_0.json'

        data_root_test = "/home/radu.berdan/datasets/NOD_h416_d32/"
        test_data_prefix = f'raw_new_{camera_name_full}_test/'
        test_ann_file = f'annotations/raw_new_{camera_name_full}_test.json'

        # SONY
        params["data_root_train"] = data_root_train
        params["train_data_prefix"] = train_data_prefix
        params["train_ann_file"] = train_ann_file

        params["data_root_test"] = data_root_test
        params["test_data_prefix"] = test_data_prefix
        params["test_ann_file"] = test_ann_file

        params["yolo_type"] = yolo_type
        norm_key = norm_link[dataset_name]

        params["preproc"] = preproc_fun

        params["preproc_params"]["name"] = preproc_fun
        params["preproc_params"]["rggb_max"] = norms[norm_key]["rggb_max"]
        params["preproc_params"]["mean"] = norms[norm_key]["mean"]
        params["preproc_params"]["std"] = norms[norm_key]["std"]

        params["rggb_max_train"] = norms[norm_key]["rggb_max"]
        params["rggb_max_test"] = norms[norm_key]["rggb_max"]

        TIME_STAMP = str(int(time.time()))

        project_name = (
            f"{TIME_STAMP}_diff_"
            f'y-{params["yolo_type"]}_'
            f'lr-{params["lr"]}_'
            f'c-{camera_name}-RAW_'
            f'r-{run_no}'
        )

        work_dir = f"./work_dirs/{project_name}/"

        params["run_path"] = TBOARD_DIR + project_name
        writer = SummaryWriter(params["run_path"])

        torch.manual_seed(0)

        cfg = Config.fromfile(CFG_PATH)
        cfg.work_dir = work_dir
        cfg = update_cfg(cfg, params)
        print('0'*200)
        runner = RUNNERS.build(cfg)

        train_dataloader = runner.train_dataloader

        runner._train_loop = runner.build_train_loop(runner._train_loop)
        runner._val_loop = runner.build_val_loop(runner._val_loop)
        runner.call_hook("before_run")
        runner._init_model_weights()
        runner.load_or_resume()
        runner.call_hook("before_train")
        runner.call_hook("before_train_epoch")

        model = runner.model

        # model.eval()
        # metrics = runner.val_loop.run()
        # exit()
        model.train()

        print("#" * 200)
        print("Starting:", project_name)

        for epoch in range(params["max_epochs"]):
            lr = linear_lr(
                epoch,
                stop=params["max_epochs_scaling"],
                lr_max=params["lr"],
                lr_min=params["lr_min"],
            )
            writer.add_scalar("lr", lr, epoch)
            optim_wrapper = runner.optim_wrapper
            optim_wrapper["optimizer"]["lr"] = lr
            optim_wrapper = runner.build_optim_wrapper(optim_wrapper)

            print(f"-> Setup new lr: {np.round(lr,5)}")

            model.train()

            with tqdm(
                train_dataloader, unit="batch", disable=DISABLE
            ) as tepoch:
                loss = torch.tensor(0)
                last_loss = 0
                for data in tepoch:
                    last_loss = last_loss * 0.9 + loss.item() * 0.1
                    tepoch.set_description(
                        f"Epoch {epoch} | loss={round(last_loss, 5):0.5f}"
                    )
                    # get the inputs; data is a list of [inputs, labels]
                    with optim_wrapper.optim_context(model):
                        data = model.data_preprocessor(data, True)
                        #print(data['inputs'][0][:,320,320])
                        losses = model._run_forward(data, mode="loss")
                        loss, log_vars = model.parse_losses(losses)

                    optim_wrapper.update_params(loss)

            model.eval()
            metrics = runner.val_loop.run()
            for metric_name in metrics.keys():
                writer.add_scalar(metric_name, metrics[metric_name], epoch)
            print(metrics)
