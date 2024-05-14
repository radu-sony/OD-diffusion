import torch
import sys
import matplotlib.pyplot as plt

# device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
# torch.cuda.set_device(device)

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
rggb_max = 2**14

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


disable = False
tboard_dir = './runs/runs-cocoraw-v1-2b-finetune/' 

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

model_path_rggb = '/home/radu.berdan/work/OD/work_dirs/1704955444_yolo-s_r-640_gamma-cnn_finetuned-rggb-gamma-0.45-8-trainable_d-cocoraw-v1-nod+nod_0/best_coco_bbox_mAP_epoch_0.pth'

# model_path = '/home/radu/work/raw/det-train-4/work_dirs/1701664552_yolo-m_norm-max_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth' # 320
# #model_path = '/home/radu/work/raw/det-train-4/work_dirs/1701757296_yolo-m_norm-max_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth' # 640

base_datasets = ['nod']
preproc_funs = ['gamma-cnn']
yolos = ['n','s','m']
splits = ['_0']

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]
params['batch_size'] = 64

cuda_no = str(sys.argv[2])
device = torch.device(f'cuda:{cuda_no}')  # GPU 1 is 'cuda:1'
torch.cuda.set_device(device)


for base_dataset in base_datasets:
    for split in splits:
        for preproc_fun in preproc_funs:
            dataset_name = f'cocoraw-v1-{base_dataset}+{base_dataset}'
            #dataset_name = base_dataset
            train_anno_file_path = f'annotations/{dataset_name}_train{split}.json'

            #train_anno_file_path = paths[base_dataset]['train_ann_file']

            model_path = model_path_rggb

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
            project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_r-{params["img_scale"][0]}_{preproc_fun}_finetuned-rggb-gamma-{gamma}-{hidden_channels}_d-{dataset_name}{split}'  
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
            
            #try:
            model.load_state_dict(torch.load(model_path)['state_dict'])
            # except:
            #     print('!!! Some size mismatch.')

            model = runner.model
            print(model)
            model_preproc = model.backbone.preproc

            print(model_preproc)
            print(model_preproc.cnn1[0].bias)
            #print(model_preproc.cnn2[0].weights)


            n = 100
            in_chans = 4

            x0 = np.linspace(0,1,n)
            x = x0
            x = np.repeat(x, in_chans).reshape(-1, in_chans)

            x = np.expand_dims(np.expand_dims(x, -1), -1)
            x = torch.from_numpy(x).cuda().float()

            y = model_preproc(x)
            print(y.shape)
            # xin = [x[i,0,0,0].item()/rggb_max for i in range(n)]
            # yout = [y[i,0,0,0].item() for i in range(n)]
            # y0 = x0**gamma
            # print(xin)
            # print(yout)
            # print(y0)
            # print(y[:,:,0,0])

            #model_preproc.init_params()

            #print(model_preproc.cnn1[0].bias)
            #print(model_preproc.cnn1[0].weight)

            #for j in range(n):
                #print([y[i,j,0,0].item() for i in range(n)])
            plt.plot(x0*rggb_max, [y[j,1,0,0].item() for j in range(n)])

            stamp = str(int(time.time()))
            plt.savefig(f'./outputs/{stamp}.jpg')

            
            model_preproc.init_params()
            print(model_preproc.cnn1[0].bias)
            #print(model_preproc.cnn2[0].weights)






        




