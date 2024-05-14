from mmengine.config import Config
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner

from mmyolo.utils import is_metainfo_lower
from resources.utils import make_name, update_cfg, cosine
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

cfg_path = './configs/yolov8_n_pascalraw.py'

params['data_root_train'] = '/home/radu/work/datasets/'
params['train_data_prefix'] = ''
params['train_ann_file'] = 'annotations/cocoraw-v1-nod+nod_train_30000.json'

params['data_root_test'] = '/home/radu/work/datasets/NOD_h400/'
params['test_data_prefix'] = 'raw/'
params['test_ann_file'] = 'annotations/raw_new_Nikon750_test_flip.json'

params['lr'] = 0.01
params['lr_min'] = 0.001
params['stop_cosine_epochs'] = 32
params['max_epochs'] = 40
params['img_scale'] = (320, 320)
params['rggb_max_train'] = 2**14
params['rggb_max_test'] = 2**14

params['dataset_type'] = 'rggb' # no preprocessing done by mm library
params['preproc'] = 'norm'

params['preproc_params'] = {'name':'norm',
                            'mean': [],
                            'std': []}

DISABLE = False
TBOARD_DIR = './runs/runs-cocoraw-v1-3-nod/' 
params["yolo_type"] = 'n'

dataset_name = 'cocoraw-v1-nod+nod'
norm_key = norm_link[dataset_name]

params['rggb_max_train'] = norms[norm_key]['rggb_max']
params['preproc_params']['mean'] = [x*norms[norm_key]['rggb_max'] for x in norms[norm_key]['mean']]
params['preproc_params']['std'] = [x*norms[norm_key]['rggb_max'] for x in norms[norm_key]['std']]

time_stamp = str(int(time.time()))

project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_e-{params["max_epochs"]}_m-{params["model_type"]}_d-{dataset_name}_3000'  
print('Starting:', project_name)

work_dir = f'./work_dirs/{project_name}/'

params['run_path'] = TBOARD_DIR + project_name
writer = SummaryWriter(params['run_path'])

torch.manual_seed(0)

cfg = Config.fromfile(cfg_path)
cfg.work_dir = work_dir
cfg = update_cfg(cfg, params)
runner = RUNNERS.build(cfg)

train_dataloader = runner.train_dataloader
runner._train_loop = runner.build_train_loop(runner._train_loop)
runner._val_loop = runner.build_val_loop(runner._val_loop)
runner.call_hook('before_run')
runner._init_model_weights()
runner.load_or_resume()
runner.call_hook('before_train')
runner.call_hook('before_train_epoch')

model = runner.model

for epoch in range(params['max_epochs']):  # loop over the dataset multiple times

    lr = cosine(epoch, params['stop_cosine_epochs'], params['lr'], params['lr_min'])
    optim_wrapper = runner.optim_wrapper
    optim_wrapper['optimizer']['lr'] = lr
    optim_wrapper = runner.build_optim_wrapper(optim_wrapper)
    print(f'-> Setup new lr: {np.round(lr,5)}')

    model.train()

    with tqdm(train_dataloader, unit='batch', disable=disable) as tepoch:
        loss = torch.tensor(0) 
        last_loss = 0
        for data in tepoch:
            last_loss = last_loss*0.9+loss.item()*0.1
            tepoch.set_description(f'Epoch {epoch} | loss={"%.5f" % round(last_loss, 5)}')
            # get the inputs; data is a list of [inputs, labels]
            with optim_wrapper.optim_context(model):
                data = model.data_preprocessor(data, True)

                losses = model._run_forward(data, mode='loss')
                loss, log_vars = model.parse_losses(losses)

            optim_wrapper.update_params(loss)

    model.eval()
    metrics = runner.val_loop.run()
    for metric_name in metrics.keys():
        writer.add_scalar(metric_name, metrics[metric_name], epoch)
    print(metrics)


        




