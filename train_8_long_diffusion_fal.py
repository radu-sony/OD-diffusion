import torch
import sys
cuda_no = str(sys.argv[2])

device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)

import time

from mmengine.config import Config
from mmyolo.registry import RUNNERS

from resources.utils_diffusion import update_cfg, linear_lr

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
params['dataset_type'] = 'NODDataset'


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
TBOARD_DIR = "./runs/runs-cocoraw-v1-2-diffusion/"

DISABLE = True

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 64

mean = (800.96014, 895.1913,  896.5624,  765.8472)
std = (782.79034, 963.45276, 963.0715,  752.7042)

mean = (600, 600, 600, 600)
std = (16384-600, 16384-600, 16384-600, 16384-600)



# size = int(sys.argv[3])
# print(fff)
# exit()


sizes = [500,1000]
fffs = np.arange(1,11)/10

for size in sizes:
    for fff in fffs:

        params['bgr_to_rgb'] = False
        params['mean'] = mean
        params['std'] = std
        params['input_channels'] = 3
        params['f'] = fff

        params["nod_data_root_train"] = '/home/radu.berdan/datasets/NOD_h416_d32/'
        params["nod_train_ann_file"] = f'annotations-3cat/raw_new_Nikon750_train_{size}_0.json'

        params["city_data_root_train"] = '/home/radu.berdan/datasets/'
        params["city_dataset_name"] = f'cityscapes_fal_processed'
        params["city_train_ann_file"] = 'annotations-3cat/train.json'

        params["data_root_test"] = '/home/radu.berdan/datasets/NOD_h416_d32/'
        params["test_data_prefix"] = 'raw_new_Nikon750_test/'
        params["test_ann_file"] = 'annotations-3cat/raw_new_Nikon750_test.json'

        params["yolo_type"] = yolo_type

        TIME_STAMP = str(int(time.time()))

        project_name = (
            f"{TIME_STAMP}_diff_"
            f'y-{params["yolo_type"]}_'
            f'lr-{params["lr"]}_'
            f'n-max_'
            f's-{size}_'
            f'f-{fff}_FAL'
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
