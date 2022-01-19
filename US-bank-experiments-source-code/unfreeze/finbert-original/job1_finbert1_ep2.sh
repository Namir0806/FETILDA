#!/bin/bash
#SBATCH --nodes=1
#SBATCH --N=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=06:00:00

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tk-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python ucb-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python mean-2-hk-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python max-2-hk-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python mean-4-1-hk-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python max-4-1-hk-finbert-bilstm-hist-1.py 510 sec7 roa 5 1 2 10e-4 &
wait
