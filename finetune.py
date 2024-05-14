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

from configs.params_init import *
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn as nn


import time
import os


import torch.nn.init as init

cfg_path = './configs/yolov8_n_pascalraw.py'

params['lr'] = 0.0001
params['lr_min'] = 0.00001
params['start_cosine_epochs'] = 28
params['stop_cosine_epochs'] = 36
params['img_scale'] = (640, 640)
params['max_epochs'] = 20
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
                            'scale_test': 1}

disable = True
tboard_dir = './runs/runs-cocoraw-v1-3b-finetune/' 

# model_paths = {'x': '/home/radu.berdan/work/OD/work_dirs/1702955112_y-x_f-norm-max_r-320_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                's': '/home/radu.berdan/work/OD/work_dirs/1702955112_y-s_f-norm-max_r-320_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                'm': '/home/radu.berdan/work/OD/work_dirs/1702955112_y-m_f-norm-max_r-320_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                'l': '/home/radu.berdan/work/OD/work_dirs/1702955112_y-l_f-norm-max_r-320_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth'}

# # # # 1x1
# model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1703062276_y-s_f-norm-max_r-640_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                'm': '/home/radu.berdan/work/OD/work_dirs/1703062276_y-m_f-norm-max_r-640_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                'l': '/home/radu.berdan/work/OD/work_dirs/1703062276_y-l_f-norm-max_r-640_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
#                'x': '/home/radu.berdan/work/OD/work_dirs/1703062276_y-x_f-norm-max_r-640_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth'}

# # NOD 640
# model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1703472112_y-s_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'm': '/home/radu.berdan/work/OD/work_dirs/1703472112_y-m_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'l': '/home/radu.berdan/work/OD/work_dirs/1703472112_y-l_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'x': '/home/radu.berdan/work/OD/work_dirs/1703472112_y-x_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth'}

# # # NOD 640 brightness adjust
# model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1703826862_y-s_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'm': '/home/radu.berdan/work/OD/work_dirs/1703826862_y-m_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'l': '/home/radu.berdan/work/OD/work_dirs/1703826862_y-l_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'x': '/home/radu.berdan/work/OD/work_dirs/1703826862_y-x_f-norm-max_r-640_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth'}

# NOD 320
# model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1703577847_y-s_f-norm-max_r-320_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'm': '/home/radu.berdan/work/OD/work_dirs/1703577847_y-m_f-norm-max_r-320_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'l': '/home/radu.berdan/work/OD/work_dirs/1703577847_y-l_f-norm-max_r-320_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
#                'x': '/home/radu.berdan/work/OD/work_dirs/1703577847_y-x_f-norm-max_r-320_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth'}

# model_paths = {'n': '/home/radu/work/raw/det-train-4/work_dirs/1702278723_yolo-n_norm-max_r-320_d-cocoraw-v1/best_coco_bbox_mAP_epoch_0.pth',
#                's': '/home/radu/work/raw/det-train-4/work_dirs/1702302883_yolo-s_norm-max_r-320_d-cocoraw-v1/best_coco_bbox_mAP_epoch_0.pth'}

# pascal 640 mosaic 1stage
model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1704964162_y-s_f-norm-max_r-640_a-1stage_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
               'm': '/home/radu.berdan/work/OD/work_dirs/1704964162_y-m_f-norm-max_r-640_a-1stage_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
               'l': '/home/radu.berdan/work/OD/work_dirs/1704964162_y-l_f-norm-max_r-640_a-1stage_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth',
               'x': '/home/radu.berdan/work/OD/work_dirs/1704964162_y-x_f-norm-max_r-640_a-1stage_t-pascal_d-cocoraw-v1-pascal+pascal/best_coco_bbox_mAP_epoch_0.pth'}

# nod 640 mosaic 1stage
model_paths = {'s': '/home/radu.berdan/work/OD/work_dirs/1705634389_y-s_f-norm-max_r-640_a-1stage_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
               'm': '/home/radu.berdan/work/OD/work_dirs/1705634389_y-m_f-norm-max_r-640_a-1stage_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
               'l': '/home/radu.berdan/work/OD/work_dirs/1705634389_y-l_f-norm-max_r-640_a-1stage_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth',
               'x': '/home/radu.berdan/work/OD/work_dirs/1705634389_y-x_f-norm-max_r-640_a-1stage_t-nod_d-cocoraw-v1-nod+nod/best_coco_bbox_mAP_epoch_0.pth'}


INIT_HEAD = False
CHANGE_STEM = False

grad_divs = {'n': 1,
             's': 1,
             'm': 2,
             'l': 2,
             'x': 4}

base_datasets = ["nod"]
preproc_funs = ["norm-max"]
yolos = ["l"]
splits = ["_0"]

yolo_type = str(sys.argv[1])
params['grad_div'] = grad_divs[yolo_type]

params['batch_size'] = 64

model_path = model_paths[yolo_type]
runs = 5
for run_no in range(runs):
    for base_dataset in base_datasets:
        for split in splits:
            for preproc_fun in preproc_funs:
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
                project_name = f'{time_stamp}_yolo-{params["yolo_type"]}_lr-{params["lr"]}_r-{params["img_scale"][0]}_{preproc_fun}_finetuned_r-{run_no}_d-{dataset_name}{split}'  
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

                h0 = model.bbox_head.head_module.cls_preds[0][2]
                h1 = model.bbox_head.head_module.cls_preds[1][2]
                h2 = model.bbox_head.head_module.cls_preds[2][2]
                # try:
                if CHANGE_STEM:
                    stem = model.backbone.stem.conv
                    model.backbone.stem.conv = nn.Conv2d(4, stem.out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False).cuda()
                model.load_state_dict(torch.load(model_path)['state_dict'])
                # except:
                #     pass
                # if INIT_HEAD:
                #     for i in range(3):
                #         init.kaiming_normal_(model.bbox_head.head_module.cls_preds[i][2].weight, mode='fan_out', nonlinearity='relu')
                #         if model.bbox_head.head_module.cls_preds[i][2].bias is not None:
                #             init.constant_(model.bbox_head.head_module.cls_preds[i][2].bias, 0.0)
                #     # model.bbox_head.head_module.cls_preds[0][2] = h0
                #     # model.bbox_head.head_module.cls_preds[1][2] = h1
                #     # model.bbox_head.head_module.cls_preds[2][2] = h2

                runner.call_hook('before_train')
                runner.call_hook('before_train_epoch')

                print('#'*200)
                print('Starting:', project_name)
                print('train:', train_anno_file_path)
                print('test: ', params['test_ann_file'])

                for epoch in range(params['max_epochs']):  # loop over the dataset multiple times
                    lr = linear_lr(epoch, stop=params['max_epochs'], lr_max=params['lr'], lr_min=params['lr_min'])
                    #lr = cosine(epoch, stop=params['max_epochs'], lr_max=params['lr'], lr_min=params['lr_min'])
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


            




