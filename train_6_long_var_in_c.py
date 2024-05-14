import torch
import sys
cuda_no = str(sys.argv[2])


device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)

import time

from mmengine.config import Config
from mmyolo.registry import RUNNERS

from resources.utils import update_cfg, linear_lr

from configs.params_init import params, norms, paths, norm_link
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from tqdm import tqdm
import numpy as np
import resources.models
import resources.transforms

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

params["lr"] = 0.01
params["lr_min"] = 0.001
params["max_epochs_scaling"] = 40
params["max_epochs"] = 40
params["start_cosine_epochs"] = 28
params["stop_cosine_epochs"] = 36
params["img_scale"] = (320, 320)

params["dataset_type"] = "rggb"  # no preprocessing done by mm library

params['preproc'] = 'preproc'

params["preproc_params"] = {
    "name": "dynamic1",
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
TBOARD_DIR = "./runs/runs-cocoraw-v1-1-long/"

DISABLE = True
CHANGE_STEM = True

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

base_datasets = ["pascal"]
preproc_funs = ['norm-max']
yolos = ["l"]
splits = [""]

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 256

in_chans = [48, 24, 32, 16]
in_chan_multiplier = int(str(sys.argv[3]))

for base_dataset in base_datasets:
    for split in splits:
        for preproc_fun in preproc_funs:
            # for in_chan in in_chans:

            dataset_name = f'cocoraw-v1-{base_dataset}+{base_dataset}'
            train_ann_file = f'annotations/{dataset_name}_train{split}.json'

            params["data_root_train"] = paths[base_dataset]["data_root_train"]
            params["train_data_prefix"] = paths[base_dataset]["train_data_prefix"]
            params["train_ann_file"] = train_ann_file

            params["data_root_test"] = paths[base_dataset]["data_root_test"]
            params["test_data_prefix"] = paths[base_dataset]["test_data_prefix"]
            params["test_ann_file"] = paths[base_dataset]["test_ann_file"]

            params["yolo_type"] = yolo_type
            norm_key = norm_link[dataset_name]

            # params["preproc_params"] = preproc_fun

            params["preproc_params"]["name"] = preproc_fun
            params["preproc_params"]["rggb_max"] = norms[norm_key]["rggb_max"]
            params["preproc_params"]["mean"] = norms[norm_key]["mean"]
            params["preproc_params"]["std"] = norms[norm_key]["std"]

            params["rggb_max_train"] = norms[norm_key]["rggb_max"]
            params["rggb_max_test"] = norms[norm_key]["rggb_max"]

            TIME_STAMP = str(int(time.time()))

            project_name = (
                f"{TIME_STAMP}_"
                f'y-{params["yolo_type"]}_'
                f"f-{preproc_fun}_"
                f'r-{params["img_scale"][0]}_'
                f'c-{in_chan_multiplier}_'
                f't-{base_dataset}_'
                f"d-{dataset_name}{split}"
            )

            work_dir = f"./work_dirs/{project_name}/"

            params["run_path"] = TBOARD_DIR + project_name
            writer = SummaryWriter(params["run_path"])

            torch.manual_seed(0)

            cfg = Config.fromfile(CFG_PATH)
            cfg.work_dir = work_dir
            cfg = update_cfg(cfg, params)

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

            stage1_chans = model.backbone.stage1[0].conv.out_channels
            stem_chans = model.backbone.stem.conv.out_channels

            if CHANGE_STEM:
                model.backbone.stem.conv = nn.Conv2d(
                    4,
                    stem_chans * in_chan_multiplier,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ).cuda()
                model.backbone.stem.bn = nn.BatchNorm2d(
                    stem_chans * in_chan_multiplier,
                    eps=0.001,
                    momentum=0.03,
                    affine=True,
                    track_running_stats=True
                ).cuda()
                model.backbone.stage1[0].conv = nn.Conv2d(
                    stem_chans * in_chan_multiplier,
                    stage1_chans,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                    bias=False,
                ).cuda()
            
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
