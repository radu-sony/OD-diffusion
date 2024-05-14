import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-6-schedule'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
ticks1 = ['n', 's', 'm', 'l']

with open(f'{save_path}{run_name}.pkl', 'rb') as f:
    runs = pkl.load(f)


# 2
fig_name = 'schedule'
fig, axs = plt.subplots(2, 3, dpi=200, figsize=(18,8))
splits = ['0', '10000', '30000']
idxs = {'0': 0, '10000': 1, '30000': 2, 'Full':3}

x = [0, 1, 2]
yolos = ['n']
# yolos = ['s']
colors = ['r', 'b', 'g', 'y']
sets = ['pascal+pascal', 'nod+nod', 'coco+pascal', 'coco+nod']

x = np.arange(8)

for m, metric_name in enumerate(metric_names):
    #for s, split in enumerate(splits):
    #for y, yolo_type in enumerate(yolos):
    pascal = []
    nod = []
    for run in runs:
        if run['params']['yolo_type'] in yolos:
            acc = np.max(run['metrics'][metric_name])
            idx = int(run['params']['id'])
            if 'pascal' in run['params']['trainset']:   
                pascal.append([idx, np.round(acc*100, 2)])
            if 'nod' in run['params']['trainset']:
                nod.append([idx, np.round(acc*100, 2)])
    
    pascal = [x[1] for x in sorted(pascal, key=lambda x: x[0])]
    nod = [x[1] for x in sorted(nod, key=lambda x: x[0])]

    print(len(pascal), len(nod))

    axs[0,m].plot(x, pascal, color=colors[0], marker='s', label='')
    axs[0,m].set_title(metric_name)
    #axs[0,m].set_xticks(x)
    #axs[0,m].set_xticklabels(ticks1)
    axs[0,m].set_xlabel('# of cocoraw-v1 images added to trainset')
    axs[0,m].legend()

    axs[1,m].plot(x, nod, color=colors[0], marker='s', label='')
    #axs[1,m].set_xticks(x)
    #axs[1,m].set_xticklabels(ticks1)
    axs[1,m].set_xlabel('# of cocoraw-v1 images added to trainset')
    axs[1,m].legend()                   
        
axs[0,0].set_ylabel('Accuracy (%)')
axs[1,0].set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(f'{save_path}{run_name}_{fig_name}.jpg')
plt.close()      




# for m, metric_name in enumerate(metric_names):
#     pascal = {}
#     nod = [0,0,0,0]

#     for run in runs:
#         if run['params']['yolo_type'] == 'l':
#             if 'pascal' in run['params']['trainset']:
#                 split = str(run['params']['split'])
#                 pas = np.max(run['metrics'][metric_name])
#                 pascal[idxs[split]] = np.round(pas*100, 2)
#             if 'nod' in run['params']['trainset']:
#                 split = str(run['params']['split'])
#                 nd = np.max(run['metrics'][metric_name])
#                 nod[idxs[split]] = np.round(nd*100, 2)

#     axs[m].plot(x, pascal, color='r', marker='s', label='pascalraw')
#     axs[m].plot(x, nod, color='b', marker='s', label='nod')
#     axs[m].set_title(metric_name)
#     axs[m].set_ylim([10,95])
#     axs[m].set_xticks(x)
#     axs[m].set_xticklabels(ticks1)
#     axs[m].set_xlabel('# of cocoraw-v1 images added to trainset')
#     axs[m].legend()






    





