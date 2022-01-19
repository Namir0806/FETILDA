#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=00:30:00   

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-bilstm-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-mean-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-max-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-bilstm-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-mean-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-max-2.py 510 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf4094-mean-2.py 4094 sec1A eps &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf4094-max-2.py 4094 sec1A eps &
wait
