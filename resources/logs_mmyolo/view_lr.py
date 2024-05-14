import matplotlib.pyplot as plt
import json
log_path = 'yolov8L_log.txt'

data = []
x = []
y = []
metrics = {"coco/bbox_mAP": [], "coco/bbox_mAP_50": [], "coco/bbox_mAP_75": []}
xmetrics = []
with open(log_path, 'r') as f:
    for line in f.readlines():
        data.append(json.loads(line))
        if isinstance(data[-1], dict):
            if 'lr' in data[-1].keys():
                x.append(data[-1]['step'])
                y.append(data[-1]['lr'])
            if "coco/bbox_mAP" in data[-1].keys():
                metrics["coco/bbox_mAP"].append(data[-1]["coco/bbox_mAP"])
                metrics["coco/bbox_mAP_50"].append(data[-1]["coco/bbox_mAP_50"])
                metrics["coco/bbox_mAP_75"].append(data[-1]["coco/bbox_mAP_75"])
                xmetrics.append(data[-2]['step'])

fig, axs = plt.subplots(2,1,dpi=200)

axs[0].plot(x,y)
axs[0].set_yscale('log')
for mname in metrics.keys():
    axs[1].plot(xmetrics, metrics[mname])
plt.savefig(log_path[:-4]+'.jpg')