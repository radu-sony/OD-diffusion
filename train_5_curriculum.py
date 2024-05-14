from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.optim import OptimWrapper, build_optim_wrapper

from mmyolo.utils import is_metainfo_lower
from resources.utils import make_name, update_cfg, cosine, cosine_var
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

params['lr'] = 0.01
params['lr_min'] = 0.001
params['start_cosine_epochs'] = 28
params['stop_cosine_epochs'] = 36
params['max_epochs'] = 40
params['img_scale'] = (320, 320)
params['rggb_max_train'] = 2**14
params['rggb_max_test'] = 2**14

params['dataset_type'] = 'rggb' # no preprocessing done by mm library

params['preproc_params'] = {'name':'dynamic1',
                            'value': 1,
                            'rggb_max': 2**14,
                            'hidden_size': 32,
                            'n_layers': 4,
                            'n_params': 3,
                            'mean': [],
                            'std': [],
                            'scale_pascal': 1}

disable = True
tboard_dir = './runs/runs-cocoraw-v1-11-curr/' 

base_datasets = ['pascal']
# splits = ['', '_10000', '_30000']
yolos = ['n']

base_datasets = ['pascal','nod']
preproc_funs = ['norm-max']
yolos = ['n','s','m']
resolutions = [320]
splits = ['_30000']

def gamma_schedule(epoch, g_min=0.2, g_max=1, e_start=10, e_stop=28):
    if epoch <= e_stop:
        gamma = (epoch)/(e_stop) * (g_max-g_min) + g_min
    else:
        gamma = g_max
    return gamma

def gamma_schedule(epoch, g_min=1, g_max=1.5, e_start=10, e_stop=28):
    if epoch <= e_stop:
        gamma = g_max - (epoch)/(e_stop) * (g_max-g_min)
    else:
        gamma = g_min
    return gamma


gamma_mins = [0.5]
g_max = 1.5

for yolo_type in yolos:
    for base_dataset in base_datasets:
        for split in splits:
            for preproc_fun in preproc_funs:
                for g_min in gamma_mins:
                    dataset_name = f'cocoraw-v1-{base_dataset}+{base_dataset}'
                    train_anno_file_path = f'annotations/{dataset_name}_train{split}.json'

                    params['data_root_train'] = paths[base_dataset]['data_root_train']
                    params['train_data_prefix'] = paths[base_dataset]['train_data_prefix']
                    params['train_ann_file'] = train_anno_file_path

                    params['data_root_test'] = paths[base_dataset]['data_root_test']
                    params['test_data_prefix'] = paths[base_dataset]['test_data_prefix']
                    params['test_ann_file'] = paths[base_dataset]['test_ann_file']

                    params["yolo_type"] = yolo_type
                    norm_key = norm_link[dataset_name]

                    params['preproc'] = preproc_fun

                    params['preproc_params']['name'] = preproc_fun
                    params['preproc_params']['rggb_max'] = norms[norm_key]['rggb_max']
                    params['preproc_params']['mean'] = norms[norm_key]['mean']
                    params['preproc_params']['std'] = norms[norm_key]['std']

                    params['rggb_max_train'] = norms[norm_key]['rggb_max']
                    params['rggb_max_test'] = norms[norm_key]['rggb_max']

                    time_stamp = str(int(time.time()))

                    #project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_e-{params["max_epochs"]}_m-{params["model_type"]}_d-{dataset_name}{split}'  
                    project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_gmax-{g_max}-{preproc_fun}_d-{dataset_name}{split}'  
                    # project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_abs0.3-{preproc_fun}_d-{dataset_name}{split}'  

                    work_dir = f'./work_dirs/{project_name}/'

                    params['run_path'] = tboard_dir + project_name
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

                    print('#'*200)
                    print('Starting:', project_name)
                    print('train:', train_anno_file_path)
                    print('test: ', params['test_ann_file'])

                    for epoch in range(params['max_epochs']):  # loop over the dataset multiple times

                        lr = cosine_var(epoch, params['start_cosine_epochs'], params['stop_cosine_epochs'], params['lr'], params['lr_min'])
                        writer.add_scalar('lr', lr, epoch)
                        optim_wrapper = runner.optim_wrapper
                        optim_wrapper['optimizer']['lr'] = lr
                        optim_wrapper = runner.build_optim_wrapper(optim_wrapper)
                        
                        gamma = gamma_schedule(epoch, g_max=g_max)
                        writer.add_scalar('gamma', gamma, epoch)

                        print(f'-> Setup new lr: {np.round(lr,5)}')
                        print(f'-> Setup new gamma: {np.round(gamma, 2)}')

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
                                    #gamma = torch.tensor(1+np.random.normal(0,std_gamma,size=(data['inputs'].shape[0],1,1,1))).cuda().float()
                                    data['inputs'] /= norms[norm_key]['rggb_max']
                                    #data['inputs'] = torch.clamp(data['inputs'], None, abss * norms[norm_key]['rggb_max'])
                                    data['inputs'] = data['inputs'] ** gamma
                                    data['inputs'] *= norms[norm_key]['rggb_max']
                                    # print(torch.max(data['inputs']))
                                    # print(data)
                                    # print(data.keys())

                                    # exit()

                                    losses = model._run_forward(data, mode='loss')
                                    loss, log_vars = model.parse_losses(losses)

                                optim_wrapper.update_params(loss)

                        model.eval()
                        metrics = runner.val_loop.run()
                        for metric_name in metrics.keys():
                            writer.add_scalar(metric_name, metrics[metric_name], epoch)
                        
                        print(metrics)


        




