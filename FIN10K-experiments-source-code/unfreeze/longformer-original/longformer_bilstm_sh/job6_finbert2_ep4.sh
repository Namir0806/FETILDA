#!/bin/bash
#SBATCH --nodes=1
#SBATCH --N=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=06:00:00

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 6e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 7e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 8e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 9e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 10e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 11e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 12e-4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python finbert-bilstm-2.py 4094 sec7 tobinq 5 3 4 13e-4 &
wait
