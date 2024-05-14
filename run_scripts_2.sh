#!/bin/bash

# Running scripts in the background using nohup

nohup python3 -u finetune_rggb.py x 4 &> ./logs/x1.out &
nohup python3 -u finetune_rgb.py x 5 &> ./logs/x2.out &
nohup python3 -u finetune.py x 6 &> ./logs/x3.out &

echo "All scripts have been started in the background."