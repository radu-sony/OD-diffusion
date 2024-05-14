import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-3b-finetune'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']

yolos  = ['s', 'm', 'l', 'x']


dataset_name = 'nod'

suffix_rgb = f'coco+{dataset_name}'
suffix_raw = f'cocoraw-v1-{dataset_name}'

fig_name = f'long_finetuned_{dataset_name}'

# ################################################################################
with open(f'/home/radu.berdan/work/OD/outputs/runs-cocoraw-v1-3b-finetune/runs-cocoraw-v1-3b-finetune_{suffix_rgb}.pkl', 'rb') as f:
    runs = pkl.load(f)
data_rgb = {}
for y in yolos:
    data_rgb[y]={}
    for metric in metric_names:
        data_rgb [y][metric] = []

for run in runs:
    y = run['params']['yolo_type']
    metrics = run['metrics']
    for metric_name in metric_names:
        #data_rgb[y][metric_name].append(np.max(metrics[metric_name]))
        data_rgb[y][metric_name].append(metrics[metric_name][-1])

# ################################################################################
with open(f'/home/radu.berdan/work/OD/outputs/runs-cocoraw-v1-3b-finetune/runs-cocoraw-v1-3b-finetune_{suffix_raw}.pkl', 'rb') as f:
    runs = pkl.load(f)
data_raw = {}
for y in yolos:
    data_raw[y]={}
    for metric in metric_names:
        data_raw[y][metric] = []

for run in runs:
    y = run['params']['yolo_type']
    metrics = run['metrics']
    for metric_name in metric_names:
        #data_raw[y][metric_name].append(np.max(metrics[metric_name]))
        data_raw[y][metric_name].append(metrics[metric_name][-1])

# ################################################################################

indices = np.arange(4)
bar_width = 0.35
if dataset_name == 'pascal':
    ylims = [[78,84.5], [95.5,98], [81,88], [20,31], [46,57.5], [87,92]]
else:
    ylims = [[35,43], [56,67], [35,45], [5,13], [34,45], [71,79]] 

x = [0,1,2,3]
for reduce_type in ['max','mean']:

    if reduce_type == 'max':
        fun = np.max
    else:
        fun = np.mean

    fig, axs = plt.subplots(2, 3, dpi=200, figsize=(14,8))

    v_rgb = {}
    for m,metric_name in enumerate(metric_names):
        v_rgb = [fun(data_rgb[y][metric_name])*100 for y in yolos]
        v_raw = [fun(data_raw[y][metric_name])*100 for y in yolos]

        r = m // 3
        c = m % 3

        axs[r,c].bar(indices - bar_width/2, v_rgb, bar_width, label='rgb')
        axs[r,c].bar(indices + bar_width/2, v_raw, bar_width, label='raw')

        # Adding labels for each bar
        for index, value in enumerate(v_rgb):
            axs[r,c].text(index - bar_width/2, np.round(value,2), str(np.round(value,2)), ha='center', fontsize=9)
        for index, value in enumerate(v_raw):
            axs[r,c].text(index + bar_width/2, np.round(value,2), str(np.round(value,2)), ha='center', fontsize=9)
        
        axs[r,c].set_ylim(ylims[m])
        axs[r,c].set_title(metric_name)
        axs[r,c].set_xticks(x)
        axs[r,c].set_xticklabels(yolos)
        axs[r,c].set_xlabel('YOLO size')
        axs[r,c].legend()
    
    axs[0,0].set_ylabel('Accuracy (%)')
    axs[1,0].set_ylabel('Accuracy (%)')


    plt.tight_layout()
    plt.savefig(f'{save_path}{run_name}_{fig_name}_{reduce_type}.jpg')
    plt.close()      
