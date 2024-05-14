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

# print(runs[0])
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
fig_name = 'split_2rows_delta_2'
fig, axs = plt.subplots(2, 3, dpi=200, figsize=(18,8))
idxs = {'0': 0, '10000': 1, '30000': 2, 'Full':3}
ticks1 = ['0', '10k', '30k', '70k']
x = [0, 1, 2, 3]
yolos = ['n', 's', 'm', 'l'][::-1]
yolos = ['n', 's', 'm'][::-1]
colors = ['r', 'b', 'g', 'y']
sets = ['pascal+pascal', 'nod+nod', 'coco+pascal', 'coco+nod']

for m, metric_name in enumerate(metric_names):
    for y, yolo_type in enumerate(yolos):
        pascal = {'rggb': [0,0,0,0], 'rgb': [0,0,0,0]}
        nod = {'rggb': [0,0,0,0], 'rgb': [0,0,0,0]}
        for run in runs:
            if run['params']['yolo_type'] == yolo_type:
                split = str(run['params']['split'])

                if 'coco+' in run['params']['trainset']:
                    key = 'rgb'
                else:
                    key = 'rggb'
                acc = np.max(run['metrics'][metric_name])
                if 'pascal' in run['params']['trainset']:   
                    pascal[key][idxs[split]] = np.round(acc*100, 2)
                if 'nod' in run['params']['trainset']:
                    nod[key][idxs[split]] = np.round(acc*100, 2)
        nod_aux = nod['rgb']
        pascal = [pascal['rgb'][i]-pascal['rggb'][i] for i in range(len(ticks1))]
        nod = [nod['rgb'][i]-nod['rggb'][i] for i in range(len(ticks1))]

        axs[0,m].plot(x, pascal, color=colors[y], marker='s', label='YOLOv8'+yolo_type)
        # if np.sum(pascal['rgb']) > 0:
        #     axs[0,m].plot(x, pascal['rgb'], color=colors[y], marker='s', label='YOLOv8'+yolo_type+'_rgb', linestyle='dashed')

        axs[0,m].set_title(metric_name)
        axs[0,m].set_xticks(x)
        axs[0,m].set_xticklabels(ticks1)
        axs[0,m].set_xlabel('# of cocoraw-v1 images added to trainset')
        axs[0,m].plot([0,3], [0,0], linestyle='dashed', color='k') 
        axs[0,m].legend()

        if np.sum(nod_aux) > 0:
            axs[1,m].plot(x, nod, color=colors[y], marker='s', label='YOLOv8'+yolo_type)
            # if np.sum(nod['rgb']) > 0:
            #     axs[1,m].plot(x, nod['rgb'], color=colors[y], marker='s', label='YOLOv8'+yolo_type+'_rgb', linestyle='dashed')
            axs[1,m].set_xticks(x)
            axs[1,m].set_xticklabels(ticks1)
            axs[1,m].set_xlabel('# of cocoraw-v1 images added to trainset')
            axs[1,m].legend() 
            axs[1,m].plot([0,3], [0,0], linestyle='dashed', color='k')                  
        
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






    





