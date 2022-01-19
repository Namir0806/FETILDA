#!/bin/bash
#SBATCH --nodes=1
#SBATCH --N=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=06:00:00

srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 6e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 7e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 8e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 9e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 11e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 12e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 120 -N 1 -n 1 python finbert-bilstm-1.py 510 sec7 mark_to_bk 5 1 2 13e-4 &
wait
