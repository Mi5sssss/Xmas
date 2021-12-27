#! /bin/bash
#SBATCH --partition gpu
#SBATCH --nodes=2

mpirun -np 2 python /home/rick/nas_rram/ofa/once-for-all/train_ofa_net.py
