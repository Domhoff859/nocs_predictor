#!/bin/bash

for x in 1 4 5 9 21 22 27; do
    python train.py 0 $x /home/domin/Documents/Datasets/tless/xyz_data_train
done