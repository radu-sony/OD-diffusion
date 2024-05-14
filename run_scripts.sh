#!/bin/bash

# Running scripts in the background using nohup

SCRIPT=train_9_long_diffusion_pre.py

nohup python3 -u $SCRIPT s 0 Nikon city &> ./logs/m1.out &
nohup python3 -u $SCRIPT s 1 Nikon bdd &> ./logs/m2.out &
nohup python3 -u $SCRIPT s 2 Sony city &> ./logs/m3.out &
nohup python3 -u $SCRIPT s 3 Sony bdd &> ./logs/m4.out &

echo $SCRIPT
echo "All scripts have been started in the background."