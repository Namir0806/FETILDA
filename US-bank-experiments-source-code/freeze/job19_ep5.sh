#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=00:30:00   

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-bilstm-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-mean-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python bert-max-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-bilstm-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-mean-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf-max-3.py 510 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf4094-mean-3.py 4094 sec1A roa &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python lf4094-max-3.py 4094 sec1A roa &
wait
