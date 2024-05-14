import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-10-norms'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
ticks1 = ['n', 's', 'm', 'l']

with open(f'{save_path}{run_name}.pkl', 'rb') as f:
    runs = pkl.load(f)


fig, axs = plt.subplots(2, 3, dpi=300, figsize=(18,8))
idxs = {'0': 0, '10000': 1, '30000': 2, 'Full':3}
slice_ = '30000'
x = [0, 1, 2, 3]
yolos = ['n', 's', 'm', 'l']

colors = ['r', 'b', 'k', 'y']

preproc_funs = ['norm', 'norm-max']
funs_names = ['mean-std-norm', 'max-norm']
styles=['solid']
splits = [30000]

fig_name = 'norms_only30k'


for m, metric_name in enumerate(metric_names):
    for p, preproc in enumerate(preproc_funs):
        for s, split in enumerate(splits):
            pascal = [0,0,0,0]
            nod = [0,0,0,0]

            for run in runs:
                for y, yolo_type in enumerate(yolos):
                    if run['params']['yolo_type'] == yolo_type and run['params']['dynamic_type'] == preproc and run['params']['split'] == split:
                        acc = np.max(run['metrics'][metric_name])
                        if 'pascal' in run['params']['trainset']:    
                            pascal[y] = acc
                        if 'nod' in run['params']['trainset']:    
                            nod[y] = acc

            axs[0,m].plot(x, pascal, color=colors[p], marker='s', label=funs_names[p]+f' split_{split}')
            axs[0,m].set_title(metric_name)
            axs[0,m].set_xlabel('yolo size')
            axs[0,m].set_xticks(x)
            axs[0,m].set_xticklabels(yolos)
            axs[0,m].legend()
            axs[1,m].plot(x, nod, color=colors[p], marker='s', label=funs_names[p]+f' split_{split}')
            axs[1,m].set_xlabel('yolo_size')
            axs[1,m].set_xticks(x)
            axs[1,m].set_xticklabels(yolos)       
            axs[1,m].legend()     
                   


save_path = f'{load_path}{run_name}/'

plt.tight_layout()
plt.savefig(f'{save_path}{run_name}_{fig_name}.jpg')
plt.close()      