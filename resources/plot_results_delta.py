import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-4'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
ticks1 = ['n', 's', 'm', 'l']

with open(f'{save_path}{run_name}_full1.pkl', 'rb') as f:
    runs = pkl.load(f)

# print(runs[0]['params'])
# exit()

# # 1
# fig_name = 'size'
# fig, axs = plt.subplots(1, 3, dpi=300, figsize=(14,4))
# idxs = {'n': 0, 's': 1, 'm':2, 'l':3}
# x = [0,1,2,3]
# for m, metric_name in enumerate(metric_names):
#     pascal = [0,0,0,0]
#     nod = [0,0,0,0]

#     for run in runs:
#         if run['params']['split'] == 'Full':
#             if 'pascal' in run['params']['trainset']:
#                 yolo_type = run['params']['yolo_type']
#                 pas = np.max(run['metrics'][metric_name])
#                 pascal[idxs[yolo_type]] = np.round(pas*100, 2)
#             if 'nod' in run['params']['trainset']:
#                 yolo_type = run['params']['yolo_type']
#                 nd = np.max(run['metrics'][metric_name])
#                 nod[idxs[yolo_type]] = np.round(nd*100, 2)

#     axs[m].plot(x, pascal, color='r', marker='s', label='pascalraw')
#     axs[m].plot(x, nod, color='b', marker='s', label='nod')
#     axs[m].set_title(metric_name)
#     axs[m].set_ylim([20,95])
#     axs[m].set_xticks(x)
#     axs[m].set_xticklabels(ticks1)
#     axs[m].set_xlabel('YOLOv8 size')
#     axs[m].legend()

# axs[0].set_ylabel('Accuracy (%)')
# plt.tight_layout()
# plt.savefig(f'{save_path}{run_name}_{fig_name}.jpg')
# plt.close() 

# 2
fig_name = 'split_2rows_delta'
fig, axs = plt.subplots(2, 3, dpi=200, figsize=(18,8))
splits = ['0', '10000', '30000']
idxs = {'0': 0, '10000': 1, '30000': 2, 'Full':3}
ticks1 = ['n', 's', 'm']
x = [0, 1, 2]
yolos = ['n', 's', 'm']
# yolos = ['s']
colors = ['r', 'b', 'g', 'y']
sets = ['pascal+pascal', 'nod+nod', 'coco+pascal', 'coco+nod']

for m, metric_name in enumerate(metric_names):
    for s, split in enumerate(splits):
    #for y, yolo_type in enumerate(yolos):
        pascal = {'rggb': [0,0,0], 'rgb': [0,0,0]}
        nod = {'rggb': [0,0,0], 'rgb': [0,0,0]}
        for run in runs:
            if str(run['params']['split']) == split and run['params']['yolo_type'] in yolos:
                yolo_type = str(run['params']['yolo_type'])

                if 'coco+' in run['params']['trainset']:
                    key = 'rgb'
                else:
                    key = 'rggb'
                acc = np.max(run['metrics'][metric_name])
                if 'pascal' in run['params']['trainset']:   
                    pascal[key][yolos.index(yolo_type)] = np.round(acc*100, 2)
                if 'nod' in run['params']['trainset']:
                    nod[key][yolos.index(yolo_type)] = np.round(acc*100, 2)

        pascal = [(pascal['rgb'][i]-pascal['rggb'][i])/pascal['rgb'][i] for i in range(len(yolos))]
        nod = [(nod['rgb'][i]-nod['rggb'][i])/nod['rgb'][i] for i in range(len(yolos))]

        axs[0,m].plot(x, pascal, color=colors[s], marker='s', label='split_'+split)
        axs[0,m].set_title(metric_name)
        axs[0,m].set_xticks(x)
        axs[0,m].set_xticklabels(ticks1)
        axs[0,m].set_xlabel('# of cocoraw-v1 images added to trainset')
        axs[0,m].legend()

        axs[1,m].plot(x, nod, color=colors[s], marker='s', label='split_'+split)
        axs[1,m].set_xticks(x)
        axs[1,m].set_xticklabels(ticks1)
        axs[1,m].set_xlabel('# of cocoraw-v1 images added to trainset')
        axs[1,m].legend()                   
        
axs[0,0].set_ylabel('Accuracy diff: RGB % - RAW %')
axs[1,0].set_ylabel('Accuracy diff: RGB % - RAW %')
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






    





