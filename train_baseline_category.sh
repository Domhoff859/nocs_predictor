#!/bin/bash

for x in {0..5}; do
    python train.py 0 $x /home/domin/Documents/Datasets/nocs/xyz_data
done