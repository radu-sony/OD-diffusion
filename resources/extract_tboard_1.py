import numpy as np 
from tensorboard.backend.event_processing import event_accumulator
import os
from os import listdir
from os.path import isfile, join, isdir
import pickle as pkl
import copy

metrics_dict = {'coco/bbox_mAP':[],
           'coco/bbox_mAP_50':[],
           'coco/bbox_mAP_75':[],
           'coco/bbox_mAP_s':[],
           'coco/bbox_mAP_m':[],
           'coco/bbox_mAP_l':[]}

run_name = 'runs-cocoraw-v1-4'
log_path = f'../runs/{run_name}/'
save_path = '../outputs/'
onlyfiles = [f for f in listdir(log_path) if isdir(join(log_path, f))]

if not os.path.exists(save_path + run_name):
    os.makedirs(save_path + run_name)

logs = []
for file_name in onlyfiles:
    event_acc = event_accumulator.EventAccumulator(log_path + file_name)
    event_acc.Reload()

    metrics = copy.deepcopy(metrics_dict)

    for m in metrics.keys():
        metrics[m] = [x.value for x in event_acc.Scalars(m)]

    file_str = file_name.split('_')

    params = {}
    params['yolo_type'] = file_str[1].split('-')[1]
    params['trainset'] = file_str[2]
    #params['resolution'] = int(file_str[2].split('-')[1])
    if len(file_str) > 3:
        params['split'] = int(file_str[-1])
    else:
        params['split'] = 'Full'
    
    logs.append({'params':params, 'metrics':metrics})


with open(f'{save_path}{run_name}/{run_name}_full1.pkl', 'wb') as f:
    pkl.dump(logs, f)

# print(event_acc)

# print(event_acc.Tags())
# print(event_acc.scalars())