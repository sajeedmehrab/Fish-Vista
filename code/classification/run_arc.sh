#!/bin/bash

#SBATCH -J WCESwinb
#SBATCH --output=/home/ksmehrab/FishDatasetTrack/Classification/sbatch_runs/slurm-%x.%j.out
#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=50:00:00 
#SBATCH --gres=gpu:1 
#SBATCH --partition=dgx_normal_q
#SBATCH --account=imageomicswithanuj

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11
module load GCC/11.2.0
conda init 
source ~/.bashrc

conda activate py39
which python

# wandb login
wandb login ce5bec055212677587ec7315d98a82ba89c1ab70

# Balanced
# python train.py --no_over --optimizer AdamW --decay 0.1 --model swin_b --dataset fishair_processed_bal --lr 3e-4 --lr_warmup 50 --cosine_annealing --batch-size 128 --name 'BAL_processed_swinb' --warm 150 --epoch 150 --imb_type 'none' --loss_type 'CE' --seed 9229 --num_workers 8 –server arc_fastscratch --wandb

# WCE
# python train.py --no_over --optimizer AdamW --decay 0.1 --model swin_b --dataset fishair_processed --lr 3e-4 --lr_warmup 50 --cosine_annealing --batch-size 128 --name 'WCE_processed_swinb' --warm 0 --epoch 150 --imb_type 'longtail' --loss_type 'CE' --cost --seed 9229 --num_workers 8 –server arc_fastscratch --wandb

# Focal loss
# python train.py --no_over --optimizer AdamW --decay 0.1 --model swin_b --dataset fishair_processed --lr 3e-4 --lr_warmup 50 --cosine_annealing --batch-size 128 --name 'Focal_processed_swinb' --warm 150 --epoch 150 --imb_type 'longtail' --loss_type 'Focal' --seed 9229 --num_workers 8 –server arc_fastscratch --wandb

