#!/bin/bash
#SBATCH --nodes=1
#SBATCH --N=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=06:00:00

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 6e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 7e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 8e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 9e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 11e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 12e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-3.py 4094 sec1A mark_to_bk 5 5 5 13e-4 &
wait
