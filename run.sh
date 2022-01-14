#!/bin/bash
# author: Rick Xie
# please directly use 'bash ./run.sh'
echo "Welcome to run Xmas!";

task=("teacher" "kernel" "depth" "expand")
for i in 0 1 2 3
do
    # python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[i]}
    mpirun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[i]}
done