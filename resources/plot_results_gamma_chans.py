import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-2d-finetune'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
baselines_mAP = [80.22, 81.94, 82.52, 83.78]
ticks1 = ['s', 'm', 'l', 'x']



fig_name = 'var_gamma_chans'

x = [0, 1, 2, 3]
yolos = ['n', 's', 'm', 'l'][::-1]
yolos = ['n', 's', 'm'][::-1]
colors = ['r', 'b', 'g', 'y']

chans = [str(x) for x in [2,4,8,16]]
gammas = [str(x) for x in [0.2, 0.4, 0.6, 0.8]]
yolos = ['s', 'm', 'l', 'x']

# ################################################################################
with open('/home/radu.berdan/work/OD/outputs/runs-cocoraw-v1-2d-finetune/runs-cocoraw-v1-2d-finetune_RGB_noaug.pkl', 'rb') as f:
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
        data_rgb[y][metric_name].append(np.max(metrics[metric_name]))

# ################################################################################
with open('/home/radu.berdan/work/OD/outputs/runs-cocoraw-v1-2d-finetune/runs-cocoraw-v1-2d-finetune_RGGB.pkl', 'rb') as f:
    runs = pkl.load(f)
data_rggb = {}
for y in yolos:
    data_rggb[y]={}
    for metric in metric_names:
        data_rggb [y][metric] = []

for run in runs:
    y = run['params']['yolo_type']
    metrics = run['metrics']
    for metric_name in metric_names:
        data_rggb[y][metric_name].append(np.max(metrics[metric_name]))


# ################################################################################
with open('/home/radu.berdan/work/OD/outputs/runs-cocoraw-v1-2d-finetune/runs-cocoraw-v1-2d-finetune.pkl', 'rb') as f:
    runs = pkl.load(f)
data = {}
for y in yolos:
    data[y]={}
    for chan in chans:
        data[y][chan]={}
        for gamma in gammas:
            data[y][chan][gamma] = {}
            for metric in metric_names:
                data[y][chan][gamma][metric] = []

for run in runs:
    y = run['params']['yolo_type']
    chan = run['params']['chans']
    gamma = run['params']['gamma']
    metrics = run['metrics']
    for metric_name in metric_names:
        data[y][chan][gamma][metric_name].append(np.max(metrics[metric_name]))
# ################################################################################


for reduce_type in ['max','mean']:

    if reduce_type == 'max':
        fun = np.max
    else:
        fun = np.mean

    fig, axs = plt.subplots(3, 4, dpi=200, figsize=(18,18))

    x = [0,1,2,3]
    colors = ['r', 'b', 'g', 'y']
    ylims=[[80,85], [95,98], [82,88]]
    for c, chan in enumerate(chans):
        for g, gamma in enumerate(gammas):
            for m, metric_name in enumerate(metric_names):
                y = [fun(data[y][chan][gamma][metric_name])*100 for y in yolos]
                axs[m,c].plot(x, y, marker='s', color=colors[g], label=f'gamma={gamma}')
                axs[m,c].legend()
                axs[m,c].set_xticks(x)
                axs[m,c].set_xticklabels(yolos)
                axs[m,c].set_ylim(ylims[m])
        axs[0,c].set_title(f'number of channels: {chan}')


    plt.tight_layout()
    plt.savefig(f'{save_path}{run_name}_{fig_name}_{reduce_type}.jpg')
    plt.close()      


    fig, axs = plt.subplots(3, 4, dpi=200, figsize=(18,18))
    xgamma = [float(x) for x in gammas]
    x = [0,1,2,3]
    colors = ['r', 'b', 'g', 'y']
    ylims=[[80,85], [95,98], [82,88]]
    for y, yolo in enumerate(yolos):
        for c, chan in enumerate(chans):
            for m, metric_name in enumerate(metric_names):

                yy = [fun(data[yolo][chan][gamma][metric_name])*100 for gamma in gammas]
                axs[m,y].plot(xgamma, yy, marker='s', color=colors[c], label=f'chan={chan}')
                axs[m,y].legend()
                axs[m,y].set_xlabel('gamma')
                # axs[m,y].set_xticklabels(gammas)
                #axs[m,c].set_xticks(yolos)
                #axs[m,y].set_ylim(ylims[m])
        axs[0,y].set_title(f'YOLOv8 {yolo}')
        for m, metric_name in enumerate(metric_names):
            axs[m,y].plot([0.2,0.8],[fun(data_rgb[yolo][metric_name])*100 for _ in range(2)], linestyle='dashed', color='k')
            axs[m,y].plot([0.2,0.8],[fun(data_rggb[yolo][metric_name])*100 for _ in range(2)], linestyle='dashed', color='b')

    for m, metric_name in enumerate(metric_names):
        axs[m,0].set_ylabel(metric_name)

    # for y, yolo in enumerate(yolos):
    #     axs[0,y].plot(xgamma, [baselines_mAP[y] for _ in xgamma], linestyle='dashed', color='k')

    plt.tight_layout()
    plt.savefig(f'{save_path}{run_name}_{fig_name}_3_{reduce_type}.jpg')
    plt.close()      
