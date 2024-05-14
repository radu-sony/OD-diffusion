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
import pickle as pkl
import matplotlib.pyplot as plt


params['preproc_params'] = {'name':'dynamic1',
                            'value': 1,
                            'rggb_max': 2**14,
                            'hidden_size': 32,
                            'n_layers': 4,
                            'n_params': 3,
                            'mean': [],
                            'scale_pascal': 4,
                            'std': []}


dynamic_fun = 'fundiv'
dynamic_funs = ['gamma', 'fundiv']

yolo_type = 'n'
base_dataset = 'pascal'
set_p = 0
colors = ['r', 'b']
split = '_0'

fig, axs = plt.subplots(1,3, dpi=300, figsize=(15,3))

bins = np.linspace(0,1,200)


for d, dynamic_fun in enumerate(dynamic_funs):


    params['preproc'] = f'dynamic-{dynamic_fun}'
    params['img_scale'] = (320, 320)
    params['dataset_type'] = 'rggb'

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
    params['preproc'] = f'dynamic-{dynamic_fun}'

    params['preproc_params']['rggb_max'] = norms[norm_key]['rggb_max']

    params['rggb_max_train'] = norms[norm_key]['rggb_max']
    params['rggb_max_test'] = norms[norm_key]['rggb_max']

    cfg_path = './configs/yolov8_n_pascalraw.py'

    # yolo n
    # model_paths = {'gamma':'/home/radu/work/raw/det-train-4/work_dirs/1700791609_yolo-n_dynamic-gamma_d-cocoraw-v1-pascal+pascal_30000/best_coco_bbox_mAP_epoch_0.pth',
    #                 'fundiv':'/home/radu/work/raw/det-train-4/work_dirs/1700794675_yolo-n_dynamic-fundiv_d-cocoraw-v1-pascal+pascal_30000/best_coco_bbox_mAP_epoch_0.pth'}

    # model_paths = {'gamma':'/home/radu/work/raw/det-train-4/work_dirs/1700797804_yolo-n_dynamic-gamma_d-cocoraw-v1-nod+nod_30000/best_coco_bbox_mAP_epoch_0.pth',
    #                 'fundiv':'/home/radu/work/raw/det-train-4/work_dirs/1700800928_yolo-n_dynamic-fundiv_d-cocoraw-v1-nod+nod_30000/best_coco_bbox_mAP_epoch_0.pth'}

    # yolo n
    model_paths = {'gamma':'/home/radu/work/raw/det-train-4/work_dirs/1701222454_yolo-n_dynamic-gamma-nobias_d-cocoraw-v1-pascal+pascal_0/best_coco_bbox_mAP_epoch_0.pth',
                    'fundiv':'/home/radu/work/raw/det-train-4/work_dirs/1701222856_yolo-n_dynamic-fundiv-nobias_d-cocoraw-v1-pascal+pascal_0/best_coco_bbox_mAP_epoch_0.pth'}

    model_paths = {'gamma':'/home/radu/work/raw/det-train-4/work_dirs/1701223288_yolo-n_dynamic-gamma-nobias_d-cocoraw-v1-nod+nod_0/best_coco_bbox_mAP_epoch_0.pth',
                    'fundiv':'/home/radu/work/raw/det-train-4/work_dirs/1701223703_yolo-n_dynamic-fundiv-nobias_d-cocoraw-v1-nod+nod_0/best_coco_bbox_mAP_epoch_0.pth'}



    model_path = model_paths[dynamic_fun]

    project_name = f'yolo-{params["yolo_type"]}_dynamic-{dynamic_fun}-2_d-{dataset_name}{split}_EVAL'

    cfg = Config.fromfile(cfg_path)
    cfg = update_cfg(cfg, params)
    save_dir = f'./work_dirs/{project_name}/'
    cfg['work_dir'] = save_dir
    runner = RUNNERS.build(cfg)

    # train_dataloader = runner.train_dataloader
    # runner._train_loop = runner.build_train_loop(runner._train_loop)
    # runner._val_loop = runner.build_val_loop(runner._val_loop)
    # runner.call_hook('before_run')
    # runner._init_model_weights()
    # runner.load_or_resume()
    # runner.call_hook('before_train')
    # runner.call_hook('before_train_epoch')

    # Hook function
    output = {}
    def get_layer_output(module, input, output_data):
        output['layer_output'] = output_data

    model = runner.model
    model.load_state_dict(torch.load(model_path)['state_dict'])

    model_preproc = model.backbone.preproc
    print(model_preproc)
    hook = model_preproc.params.mlp_layers[3].register_forward_hook(get_layer_output)

    train_dataloader = runner.train_dataloader


    model_preproc.eval()

    outputs = [[],[],[]]
    max_counter = 100
    counter = 0
    with tqdm(train_dataloader, unit='batch', disable=False) as tepoch:
        for data in tepoch:
            # get the inputs; data is a list of [inputs, labels]
            data = model.data_preprocessor(data, True)
            out = model_preproc(data['inputs'])
            layer_output = output['layer_output']
            layer_output = layer_output.detach().cpu().numpy()
            for i in range(3):
                for j in range(layer_output.shape[0]):
                    outputs[i].append(layer_output[j,i])
            print(layer_output)
            counter += 1
            if counter == max_counter:
                break

    hook.remove()

    print(out.size())
    print(layer_output)

    save_path = '/home/radu/work/raw/det-train-4/outputs/runs-cocoraw-v1-9-dynamic-new/'

    with open(save_path + f'{yolo_type}_{base_dataset}_{dynamic_fun}.pkl', 'wb') as f:
        pkl.dump(outputs, f)

    ######


    lists = outputs

    save_file = ''

    params_names = ['param', 'mean', 'std']


    for i in range(3):

        axs[i].hist(lists[i], bins=bins, color=colors[d], label=dynamic_fun)  # Adjust the number of bins as needed
        axs[i].set_title(params_names[i])
        axs[i].legend()

plt.savefig(save_path + f'{yolo_type}_{base_dataset}_{dynamic_fun}_2.jpg')
