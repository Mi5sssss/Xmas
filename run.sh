#!/bin/bash
# author: Rick Xie
# please directly use 'bash ./run.sh'
echo "Welcome to run Xmas!";

task=("teacher" "kernel" "depth" "expand")；
for i in 0 1 2 3
do
    # horovodrun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[i]}
    # nohup mpirun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[0]} > ./tmp/log_${task[0]} 2>&1 &
    # nohup mpirun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[1]} > ./tmp/log_${task[1]} 2>&1 &
    # nohup mpirun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[2]} > ./tmp/log_${task[2]} 2>&1 &
    # nohup mpirun -np 2 python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[3]} > ./tmp/log_${task[3]} 2>&1 &
    # nohup python ./once-for-all/train_ofa_net_cifar10_resnet.py --task ${task[i]} > ./tmp/log_${task[i]} 2>&1 &;
done