import numpy as np
import pickle as pkl
import csv
import matplotlib.pyplot as plt

load_path = '../outputs/'
run_name = 'runs-4-diffusion-pre-ours/'
save_path = f'{load_path}{run_name}/'

metric_names = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']

yolos  = ['s', 'm', 'l', 'x']
sizes = ['1000', '500']

dataset_name = 'nod'

fig_name = f'diff-100'

cameras = ['Sony', 'Nikon']
datanames = ['city', 'bdd']

# ################################################################################
with open(f'/home/radu.berdan/work/OD-diffusion/outputs/{run_name}diff-pre-ours.pkl', 'rb') as f:
    runs = pkl.load(f)
data = {}
for camera in cameras:
    data[camera] = {}
    for dataname in datanames:
        data[camera][dataname] = {}
        for metric in metric_names:
            data[camera][dataname][metric] = {'0.1':[], '0.05':[], '0.0':[], '1.0':[]}

for run in runs:
    camera = run['params']['camera']
    dataname = run['params']['dataset']
    f = run['params']['f']
    metrics = run['metrics']

    for metric_name in metric_names:
        data[camera][dataname][metric_name][f].append(np.round(metrics[metric_name][-1]*100,2))
# ################################################################################

print(data)
# exit()

camera = 'Sony'

# for camera in cameras:
#     for dataname in datanames:
#         for metric in metric_names:
#             for n_f, f in enumerate(['0.1', '0.05', '0.0']):
#                 # print(f)
#                 # print(data[camera][dataname][metric][f])
#                 mean = np.mean(data[camera][dataname][metric][f])
#                 std = np.std(data[camera][dataname][metric][f])
#                 data[camera][dataname][metric_name][f] = [mean, std]
#                 print(data[camera][dataname][metric_name][f])


lst = ['0.1', '0.05']
lst = ['1.0', '0.05', '0.0']

filename = save_path + "diffusion-yolov8s-pre-ours.csv"
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for camera in cameras:
        writer.writerow([camera])
        writer.writerow([])
        for n_f, f in enumerate(lst):
            row = [f] + metric_names
            writer.writerow(row)
            
            for dataname in datanames:
                row = [dataname]
                for metric_name in metric_names:
                    print(data[camera][dataname][metric_name][f])
                    mean = np.mean(data[camera][dataname][metric_name][f])
                    std = np.std(data[camera][dataname][metric_name][f])
                    val = str(np.round(mean,1)) + '\std{' + str(np.round(std,1))+'}'

                    row.append(val)

                writer.writerow(row)
                
            writer.writerow([])
            writer.writerow([])
        





# indices = np.arange(4)
# x = np.arange(1,11)/10
# colors=['r','b']
# fig, axs = plt.subplots(3,4, figsize=(18,10), dpi=250)

# ylims = [[8,28], [19,52], [6,28]]

# for y, yolo in enumerate(yolos):
#     for m, metric_name in enumerate(metric_names):
#         for s, size in enumerate(sizes):
#             run = data[yolo][size][metric_name]
#             run_fal = data_fal[yolo][size][metric_name]
#             axs[m,y].plot(x, run, color=colors[s], label=size, marker='s')
#             axs[m,y].plot(x, run_fal, color=colors[s], label=size, marker='s', linestyle='dashed')
#             for idx, xx in enumerate(x):
#                 axs[m,y].text(xx, run[idx]*1.01, str(run[idx]), ha='center', fontsize=9)

#         axs[m,y].legend()
#         axs[m,0].set_ylabel(f'{metric_name} (%)')
#         #axs[m,y].set_ylim(ylims[m])
#     axs[0,y].set_title(f'YOLOv8_{yolo}')

# for i in range(4):
#     axs[-1,i].set_xlabel('f')


# plt.tight_layout()
# plt.savefig(f'{save_path}{fig_name}_FAL.jpg')
# plt.close()    


