import torch
import sys
cuda_no = str(sys.argv[2])

device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmyolo.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.optim import OptimWrapper, build_optim_wrapper

from mmyolo.utils import is_metainfo_lower
from resources.utils import make_name, update_cfg, cosine, cosine_var, linear_lr
from resources import transforms
from resources import models
from resources.utils import make_cnn_gamma

from configs.params_init import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import time
import os
import sys

cfg_path = './configs/yolov8_n_pascalraw.py'

params['lr'] = 0.001
params['lr_min'] = 0.0001
params['start_cosine_epochs'] = 28
params['stop_cosine_epochs'] = 36
params['img_scale'] = (640, 640)
params['max_epochs'] = 20
params['rggb_max_train'] = 2**14
params['rggb_max_test'] = 2**14

params['dataset_type'] = 'rggb' # no preprocessing done by mm library
params['model_type'] = 'rggb' # no preprocessing done by mm library
params['preproc'] = 'norm-max'

params["num_classes"] = 3

hidden_channels = 8
gamma = 0.45

trainable = False

params['preproc_params'] = {'name':'dynamic1',
                            'value': 1,
                            'rggb_max': 2**14,
                            'hidden_size': 32,
                            'n_layers': 4,
                            'n_params': 3,
                            'mean': [],
                            'std': [],
                            'scale_test': 1,
                            'gamma': gamma,
                            'trainable': trainable,
                            'hidden_channels': hidden_channels,
                            'input_channels': 4}

disable = True
tboard_dir = './runs/runs-cocoraw-v1-2d-finetune/' 

model_paths = {'n':'/home/radu/work/models/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth',
               's':'/home/radu/work/models/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth',
               'm':'/home/radu/work/models/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth',
               'l':'/home/radu/work/models/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth',
               'x':'/home/radu/work/models/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'}

model_paths_rggb = {'n':'/home/radu.berdan/work/models/rggb/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb-rggb.pth',
                    's':'/home/radu.berdan/work/models/rggb/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1-rggb.pth',
                    'm':'/home/radu.berdan/work/models/rggb/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a-rggb.pth',
                    'l':'/home/radu.berdan/work/models/rggb/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6-rggb.pth',
                    'x':'/home/radu.berdan/work/models/rggb/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c-rggb.pth'}

# model_path = '/home/radu/work/raw/det-train-4/work_dirs/1701664552_yolo-m_norm-max_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth' # 320
# #model_path = '/home/radu/work/raw/det-train-4/work_dirs/1701757296_yolo-m_norm-max_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth' # 640

base_datasets = ['pascal']
preproc_funs = ['gamma-cnn']
yolos = ['n','s','m']
splits = ['_0']

if trainable:
    grad_divs = {'n': 2,
                's': 2,
                'm': 4,
                'l': 4,
                'x': 8}
else:
    grad_divs = {'n': 1,
                's': 1,
                'm': 2,
                'l': 2,
                'x': 4}

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 64

chans = str(sys.argv[3])

hidden_channels_arr = [int(chans)]
gamma_arr = [0.2, 0.4, 0.6, 0.8]
runs = 5

for run_no in range(runs):
    for hidden_channels in hidden_channels_arr:
        for gamma in gamma_arr:
            params['preproc_params']['gamma'] = gamma
            params['preproc_params']['hidden_channels'] = hidden_channels

            for base_dataset in base_datasets:
                for split in splits:
                    for preproc_fun in preproc_funs:
                        dataset_name = f'cocoraw-v1-{base_dataset}+{base_dataset}'
                        #dataset_name = base_dataset
                        train_anno_file_path = f'annotations/{dataset_name}_train{split}.json'

                        #train_anno_file_path = paths[base_dataset]['train_ann_file']

                        model_path = model_paths_rggb[yolo_type]

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
                        project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_r-{params["img_scale"][0]}_{preproc_fun}_finetuned-rggb-gamma-{gamma}-{hidden_channels}_run-{run_no}_d-{dataset_name}{split}'  
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
                        model = runner.model
                        try:
                            model.load_state_dict(torch.load(model_path)['state_dict'])
                        except:
                            print('!!! Some size mismatch.')
                        runner.call_hook('before_train')
                        runner.call_hook('before_train_epoch')

                        model = runner.model

                        model.backbone.preproc.init_params()

                        print('#'*200)
                        print('Starting:', project_name)
                        print('train:', train_anno_file_path)
                        print('test: ', params['test_ann_file'])

                        for epoch in range(params['max_epochs']):  # loop over the dataset multiple times

                            lr = linear_lr(epoch, stop=params['max_epochs'], lr_max=params['lr'], lr_min=params['lr_min'])
                            writer.add_scalar('lr', lr, epoch)
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


        




