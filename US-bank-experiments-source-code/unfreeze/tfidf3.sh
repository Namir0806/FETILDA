#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
##SBATCH -p debug
#SBATCH --time=06:00:00
##SBATCH --time=00:30:00   

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A Z_score_c hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A mark_to_bk hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 Z_score_c hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 mark_to_bk hist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A Z_score_c nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec1A mark_to_bk nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 Z_score_c nohist &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python tfidf_reg_org.py sec7 mark_to_bk nohist &
wait