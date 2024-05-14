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

model_paths = {'n':'/home/radu.berdan/work/models/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth',
               's':'/home/radu.berdan/work/models/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth',
               'm':'/home/radu.berdan/work/models/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth',
               'l':'/home/radu.berdan/work/models/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth',
               'x':'/home/radu.berdan/work/models/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'}

params["lr"] = 0.001
params["lr_min"] = 0.0001
params["max_epochs_scaling"] = 10
params["max_epochs"] = 10
params["start_cosine_epochs"] = 28
params["stop_cosine_epochs"] = 36
params["img_scale"] = (640, 416)

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
TBOARD_DIR = "./runs/runs-4-diffusion-pre-RGB/"

DISABLE = False
CHANGE_STEM = False

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

base_datasets = ["nod"]
preproc_funs = ['norm-max']
yolos = ["l"]
splits = [""]

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 64

base_dataset = 'nod'
runs = 3

model_path = model_paths[yolo_type]

cameras = ['Nikon750', 'Sony_RX100m7']

for n_run in range(runs):
    for camera in cameras:

        for split in splits:
            for preproc_fun in preproc_funs:

                params["data_root_train"] = '/home/radu.berdan/datasets/NOD_h416_d32/'
                params["train_data_prefix"] = f'rawpy_new_{camera}_train/'
                params["train_ann_file"] = f'annotations/rawpy_new_{camera}_train_100_0.json'

                params["data_root_test"] = '/home/radu.berdan/datasets/NOD_h416_d32/'
                params["test_data_prefix"] = f'rawpy_new_{camera}_test/'
                params["test_ann_file"] = f'annotations/rawpy_new_{camera}_test.json'

                params["yolo_type"] = yolo_type
                # norm_key = norm_link[dataset_name]

                # params["preproc_params"] = preproc_fun

                params["preproc_params"]["name"] = preproc_fun
                params["preproc_params"]["rggb_max"] = 2**8
                params["preproc_params"]["mean"] = [0,0,0]
                params["preproc_params"]["std"] = [1,1,1]

                params["rggb_max_train"] = 2**8
                params["rggb_max_test"] = 2**8


                TIME_STAMP = str(int(time.time()))

                project_name = (
                    f"{TIME_STAMP}_"
                    f'y-{params["yolo_type"]}_'
                    f"f-{preproc_fun}_"
                    f'r-{params["img_scale"][0]}_'
                    f't-{base_dataset}-{camera.replace("_","-")}_'
                    f"r-{n_run}"
                )

                work_dir = f"./work_dirs/{project_name}/"

                params["run_path"] = TBOARD_DIR + project_name
                writer = SummaryWriter(params["run_path"])

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
                #model = runner.model
                try:
                    runner.model.load_state_dict(torch.load(model_path)['state_dict'])
                except:
                    print('Some mismatch.')
                runner.call_hook("before_train")
                runner.call_hook("before_train_epoch")

                model = runner.model

                model.train()
                # metrics = runner.val_loop.run()

                # exit()

                if CHANGE_STEM:
                    # stem = model.backbone.stem.conv
                    # model.backbone.stem.conv = nn.Conv2d(
                    #     4,
                    #     stem.out_channels,
                    #     kernel_size=(1, 1),
                    #     stride=(1, 1),
                    #     padding=(0, 0),
                    #     bias=False,
                    # ).cuda()

                    model.backbone.stem.conv = nn.Conv2d(
                        4,
                        stem.out_channels,
                        kernel_size=(2, 2),
                        stride=(2, 2),
                        padding=(0, 0),
                        bias=False,
                    ).cuda()

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
