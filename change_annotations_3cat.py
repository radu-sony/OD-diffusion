import json
import numpy as np

from os import listdir
from os.path import isfile, join


mypath = '/home/radu.berdan/datasets/diffusion_NOD_h416_d32_Nikon750_1000_0_Cityscapes_h416_ddim24_bl/annotations/'
destination = '/home/radu.berdan/datasets/diffusion_NOD_h416_d32_Nikon750_1000_0_Cityscapes_h416_ddim24_bl/annotations-3cat/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

cats = [{'supercategory': 'none', 'id': 0, 'name': 'person'}, {'supercategory': 'none', 'id': 1, 'name': 'bicycle'}, {'supercategory': 'none', 'id': 2, 'name': 'car'}]

for file_name in onlyfiles:
    with open(mypath+file_name, 'rb') as f:
        annos = json.load(f)
    print(annos.keys())
    annos['categories'] = cats
    json_object = json.dumps(annos, indent=4)
    with open(destination+file_name, 'w') as f :
        f.write(json_object)
    
    # with open(destination+file_name, 'rb') as f:
    #     annos = json.load(f)
    
    # print(annos['categories'])

