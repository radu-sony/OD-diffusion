from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.optim import OptimWrapper, build_optim_wrapper

from mmyolo.utils import is_metainfo_lower
from resources.utils import make_name, update_cfg, cosine, update_testset
from resources import transforms
from resources import models

from configs.params_init import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from torchsummary import summary

import time
import os
import sys

work_dir = '1702368394_yolo-n_norm-max_r-320_s-1x1_d-cocoraw-v1-pascal+pascal'
model_path = f'/home/radu/work/raw/det-train-4/work_dirs/{work_dir}/best_coco_bbox_mAP_epoch_0.pth'
cfg_path = f'/home/radu.berdan/work/OD/work_dirs/{work_dir}/yolov8_n_pascalraw.py'

params['data_root_test'] = '/home/radu/work/datasets/pascalraw_h400a/'
params['test_data_prefix'] = 'raw/'
params['test_ann_file'] = 'annotations/flip_dng_instances_test.json'
 
cfg = Config.fromfile(cfg_path)
cfg = update_testset(cfg, params)
runner = RUNNERS.build(cfg)

runner._train_loop = runner.build_train_loop(runner._train_loop)
runner._val_loop = runner.build_val_loop(runner._val_loop)
runner.call_hook('before_run')
runner.load_or_resume()
model = runner.model
print(model.backbone.stem)
# print(model.bbox_head.head_module.cls_preds)
# print('model.bbox_head.head_module.cls_preds')
# model.load_state_dict(torch.load(model_path_2)['state_dict'])

model.backbone.stem.conv = nn.Conv2d(4, 32, kernel_size=(1,1), stride=(1, 1), padding=(0, 0), bias=False)

model.eval()

print(model.backbone.stem)


# print(model.backbone.stem)

# stem = model.backbone.stem
# stage1 = model.backbone.stage1

# stem.conv = nn.Conv2d(4, 32, kernel_size=(3,3), stride=(2, 2), padding=(1, 1), bias=False)
# model.backbone.stem = stem

# stage1[0].conv = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2, 2), padding=(1, 1), bias=False)
# model.backbone.stage1 = stage1
# # model.eval()

# print(model.backbone.stem)
# print(model.backbone.stage1)


# metrics = runner.val_loop.run()
# print(metrics)
