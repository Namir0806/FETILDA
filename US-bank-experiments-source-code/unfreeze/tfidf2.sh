#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=00:30:00   

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A leverage hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A tier1_c hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 leverage hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 tier1_c hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A leverage nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A tier1_c nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 leverage nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 tier1_c nohist &
wait