#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=03:00:00

srun --gres=gpu:4 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-bilstm-1.py 510 sec7 roa &
srun --gres=gpu:4 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-bilstm-hist-1.py 510 sec7 roa &