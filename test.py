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

from torchsummary import summary

import time
import os
import sys

model_path = '/home/radu/work/raw/det-train-4/work_dirs/1699969832_yolo-n_e-48_m-rggb_normed_cocoraw-v1-pascal/best_coco_bbox_mAP_epoch_0.pth'
cfg_path = '/home/radu/work/raw/det-train-4/work_dirs/1699969832_yolo-n_e-48_m-rggb_normed_cocoraw-v1-pascal/yolov8_n_pascalraw.py'

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
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()

metrics = runner.val_loop.run()
print(metrics)
