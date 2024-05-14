import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-cocoraw-v1-2-diffusion/'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']#, 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']

yolos  = ['s', 'm', 'l', 'x']
sizes = ['1000', '500']

dataset_name = 'nod'

fig_name = f'cityscapes_500_1000'

# ################################################################################
with open(f'/home/radu.berdan/work/OD-diffusion/outputs/runs-cocoraw-v1-2-diffusion/cityscapes_500_1000.pkl', 'rb') as f:
    runs = pkl.load(f)
data = {}
for y in yolos:
    data[y] = {}
    for s in sizes:
        data[y][s] = {}
        for metric in metric_names:
            data[y][s][metric] = [0 for _ in range(10)]

for run in runs:
    y = run['params']['yolo_type']
    f = run['params']['f']
    s = run['params']['size']
    metrics = run['metrics']
    idx = int(float(f)*10) - 1
    for metric_name in metric_names:
        data[y][s][metric_name][idx] = np.round(metrics[metric_name][-1]*100,2)
# ################################################################################



# ################################################################################
with open(f'/home/radu.berdan/work/OD-diffusion/outputs/runs-cocoraw-v1-2-diffusion/cityscapes_500_1000_FAL.pkl', 'rb') as f:
    runs = pkl.load(f)
data_fal = {}
for y in yolos:
    data_fal[y] = {}
    for s in sizes:
        data_fal[y][s] = {}
        for metric in metric_names:
            data_fal[y][s][metric] = [0 for _ in range(10)]

for run in runs:
    y = run['params']['yolo_type']
    f = run['params']['f']
    s = run['params']['size']
    metrics = run['metrics']
    idx = int(float(f)*10) - 1
    for metric_name in metric_names:
        data_fal[y][s][metric_name][idx] = np.round(metrics[metric_name][-1]*100,2)
# ################################################################################


indices = np.arange(4)
x = np.arange(1,11)/10
colors=['r','b']
fig, axs = plt.subplots(3,4, figsize=(18,10), dpi=250)

ylims = [[8,28], [19,52], [6,28]]

for y, yolo in enumerate(yolos):
    for m, metric_name in enumerate(metric_names):
        for s, size in enumerate(sizes):
            run = data[yolo][size][metric_name]
            run_fal = data_fal[yolo][size][metric_name]
            axs[m,y].plot(x, run, color=colors[s], label=size, marker='s')
            axs[m,y].plot(x, run_fal, color=colors[s], label=size, marker='s', linestyle='dashed')
            for idx, xx in enumerate(x):
                axs[m,y].text(xx, run[idx]*1.01, str(run[idx]), ha='center', fontsize=9)

        axs[m,y].legend()
        axs[m,0].set_ylabel(f'{metric_name} (%)')
        #axs[m,y].set_ylim(ylims[m])
    axs[0,y].set_title(f'YOLOv8_{yolo}')

for i in range(4):
    axs[-1,i].set_xlabel('f')


plt.tight_layout()
plt.savefig(f'{save_path}{fig_name}_FAL.jpg')
plt.close()    


# for reduce_type in ['max','mean']:

#     if reduce_type == 'max':
#         fun = np.max
#     else:
#         fun = np.mean

#     fig, axs = plt.subplots(2, 3, dpi=200, figsize=(14,8))

#     v_rgb = {}
#     for m,metric_name in enumerate(metric_names):
#         v_rgb = [fun(data_rgb[y][metric_name])*100 for y in yolos]
#         v_raw = [fun(data_raw[y][metric_name])*100 for y in yolos]

#         r = m // 3
#         c = m % 3

#         axs[r,c].bar(indices - bar_width/2, v_rgb, bar_width, label='rgb')
#         axs[r,c].bar(indices + bar_width/2, v_raw, bar_width, label='raw')

#         # Adding labels for each bar
#         for index, value in enumerate(v_rgb):
#             axs[r,c].text(index - bar_width/2, np.round(value,2), str(np.round(value,2)), ha='center', fontsize=9)
#         for index, value in enumerate(v_raw):
#             axs[r,c].text(index + bar_width/2, np.round(value,2), str(np.round(value,2)), ha='center', fontsize=9)
        
#         axs[r,c].set_ylim(ylims[m])
#         axs[r,c].set_title(metric_name)
#         axs[r,c].set_xticks(x)
#         axs[r,c].set_xticklabels(yolos)
#         axs[r,c].set_xlabel('YOLO size')
#         axs[r,c].legend()
    
#     axs[0,0].set_ylabel('Accuracy (%)')
    # axs[1,0].set_ylabel('Accuracy (%)')
  
