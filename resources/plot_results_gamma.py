import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-7-gamma-test'
run_name = 'runs-cocoraw-v1-7-ag-varnorm'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
ticks1 = ['n', 's', 'm', 'l']

with open(f'{save_path}{run_name}_fundiv-gamma.pkl', 'rb') as f:
    runs = pkl.load(f)



# 2
fig_name = '2rows_ag_varnorm'
fig, axs = plt.subplots(2, 3, dpi=200, figsize=(18,8))
idxs = {'0': 0, '10000': 1, '30000': 2, 'Full':3}
ticks1 = ['0', '10k', '30k', '70k']
x = [0, 1, 2, 3]
yolos = ['n', 's', 'm', 'l'][::-1]
yolos = ['n', 's', 'm'][::-1]
colors = ['r', 'b', 'g', 'y']
sets = ['pascal+pascal', 'nod+nod', 'coco+pascal', 'coco+nod']

preproc_funs = ['g', 'a']
markers = ['s', 'o']

gammas = []
linestyles = ['solid', 'dashed']


for m, metric_name in enumerate(metric_names):
    for y, yolo_type in enumerate(yolos):
        for p, preproc in enumerate(preproc_funs):
            pascal = []
            nod = []
            for run in runs:
                if run['params']['yolo_type'] == yolo_type and run['params']['param_type'] == preproc:
                    split = str(run['params']['split'])
                    acc = np.max(run['metrics'][metric_name])
                    gamma = float(run['params']['param'])
                    if 'pascal' in run['params']['trainset']:   
                        pascal.append([gamma, np.round(acc*100, 2)])
                    if 'nod' in run['params']['trainset']:
                        nod.append([gamma, np.round(acc*100, 2)])
            if len(pascal) > 0:
                xp = [x[0] for x in sorted(pascal, key=lambda x: x[0])]
                xn = [x[0] for x in sorted(nod, key=lambda x: x[0])]
                pascal = [x[1] for x in sorted(pascal, key=lambda x: x[0])]
                nod = [x[1] for x in sorted(nod, key=lambda x: x[0])]

                axs[0,m].plot(xp, pascal, color=colors[y], marker='s', linestyle=linestyles[p], label='YOLOv8'+yolo_type+f'_{preproc}')
                axs[0,m].set_title(metric_name)
                axs[0,m].set_xlabel('static gamma')
                axs[0,m].legend()

                axs[1,m].plot(xn, nod, color=colors[y], marker='s', linestyle=linestyles[p], label='YOLOv8'+yolo_type+f'_{preproc}')
                axs[1,m].set_xlabel('static gamma')
                axs[1,m].legend()              
        
axs[0,0].set_ylabel('Accuracy %')
axs[1,0].set_ylabel('Accuracy %')
plt.tight_layout()
plt.savefig(f'{save_path}{run_name}_{fig_name}.jpg')
plt.close()      