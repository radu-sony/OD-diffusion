import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-5-resolution'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
metric_names2 = ['coco/bbox_mAP_l', 'coco/bbox_mAP_m', 'coco/bbox_mAP_s']
ticks1 = ['n', 's', 'm', 'l']

with open(f'{save_path}{run_name}.pkl', 'rb') as f:
    runs = pkl.load(f)

fig_name = 'resolution2'
fig, axs = plt.subplots(2, 3, dpi=300, figsize=(18,8))

x = [320, 416, 512, 608]
for m, metric_name in enumerate(metric_names):
    for split in [0, 10000, 30000]:
        acc = {}
        for run in runs:
            if run['params']['split'] == split:
                acc[str(run['params']['resolution'])] = np.max(run['metrics'][metric_name])
        y = [np.round(100*acc[str(res)],2) for res in x]
        print(y)
        axs[0,m].plot(x, y, label=f'slice-{split}', marker='s')
    axs[0,m].legend()
    axs[0,m].set_title(metric_name)
    # axs[m].set_ylim([50,95])
    axs[0,m].set_xlabel('Input resolution')

x = [320, 416, 512, 608]
for m, metric_name in enumerate(metric_names2):
    for split in [0, 10000, 30000]:
        acc = {}
        for run in runs:
            if run['params']['split'] == split:
                acc[str(run['params']['resolution'])] = np.max(run['metrics'][metric_name])
        y = [np.round(100*acc[str(res)],2) for res in x]
        print(y)
        axs[1,m].plot(x, y, label=f'slice-{split}', marker='s')
    axs[1,m].legend()
    axs[1,m].set_title(metric_name)
    # axs[m].set_ylim([50,95])
    axs[1,m].set_xlabel('Input resolution')
     

axs[0,0].set_ylabel('Accuracy (%)')
axs[1,0].set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(f'{save_path}{run_name}_{fig_name}.jpg')
plt.close() 




    





