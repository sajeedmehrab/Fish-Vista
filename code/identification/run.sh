#!/bin/bash

#SBATCH -J RRresnext
#SBATCH --output=/home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Slurms/slurm-%x.%j.out
#SBATCH --cpus-per-task=8 # this requests 1 node, 8 core. 
#SBATCH --time=35:00:00 
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

# MODEL_NAME=resnet18
# MODEL_NAME=resnet34
# MODEL_NAME=resnet50

# python train.py --lr 3e-4 --lr_warmup 50 --model vit_b_32 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vit32_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_vit32_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --lr 3e-4 --lr_warmup 50 --model vit_b_16 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vit16_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_vit16_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --lr 3e-4 --lr_warmup 50 --model swin_b --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_swinb_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_swinb_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --lr 1e-4 --lr_warmup 5 --model vgg19 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vgg19_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_vgg19_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --lr 1e-4 --lr_warmup 5 --model resnet18 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_r18_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_r18_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/inception_v3_wbce_basic/ckpt_9229_S9229_tid_iv3_basic_fishair_processed_inception_v3.t7 --lr 1e-4 --lr_warmup 5 --model inception_v3 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/inception_v3_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_iv3_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/convnext_base_wbce_basic/ckpt_9229_S9229_tid_convnext_base_basic_fishair_processed_convnext_base.t7 --lr 1e-4 --lr_warmup 5 --model convnext_base --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/convnext_base_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_convnext_base_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/efficientnet_v2_m_wbce_basic/ckpt_9229_S9229_tid_efficientnet_v2_m_basic_fishair_processed_efficientnet_v2_m.t7 --lr 1e-4 --lr_warmup 5 --model efficientnet_v2_m --batch-size 32 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/efficientnet_v2_m_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_efficientnet_v2_m_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/maxvit_t_wbce_basic/ckpt_9229_S9229_tid_maxvit_t_basic_fishair_processed_maxvit_t.t7 --lr 1e-4 --lr_warmup 50 --model maxvit_t --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/maxvit_t_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_maxvit_t_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/cvt_13_wbce_basic/ckpt_9229_S9229_tid_cvt_13_basic_fishair_processed_cvt_13.t7 --lr 1e-4 --lr_warmup 50 --model cvt_13 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/cvt_13_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name tid_cvt_13_basic --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/mobilenet_v3_large_wbce_basic/ckpt_9229_S9229_tid_mobilenet_v3_large_basic_fishair_processed_mobilenet_v3_large.t7 --lr 1e-4 --lr_warmup 50 --model mobilenet_v3_large --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/mobilenet_v3_large_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name mobilenet_v3_large --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

python train.py --resume --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/resnext50_32x4d_wbce_basic/ckpt_9229_S9229_tid_resnext50_32x4d_basic_fishair_processed_resnext50_32x4d.t7 --lr 1e-4 --lr_warmup 50 --model resnext50_32x4d --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/resnext50_32x4d_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name resnext50_32x4d --loss_type WBCE --wandb --cosine_annealing --num_workers 8 --server arc

# python train.py --lr 3e-4 --lr_warmup 50 --model swinb_22k --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_swinb_22k_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name swinb_22k_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# python train.py --lr 1e-4 --lr_warmup 50 --model mobile_vit_xs --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_mobile_vit_xs_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name mobile_vit_xs_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# python train.py --lr 1e-4 --lr_warmup 50 --model mobile_vit_v2 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_mobile_mobile_vit_v2_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name mobile_vit_v2_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# python train.py --lr 1e-4 --lr_warmup 5 --model regnet_y --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_regnet_y_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name regnet_y_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# python train.py --lr 1e-4 --lr_warmup 50 --model diet_distilled_s --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_diet_distilled_s_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name diet_distilled_s_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# python train.py --lr 1e-4 --lr_warmup 50 --model pvt_v2 --batch-size 128 --epoch 100 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_pvt_v2_wbce_basic --seed 9229 --dataset fishair_processed --optimizer AdamW --decay 0.1 --name pvt_v2_wbce_basic --loss_type WBCE --cosine_annealing --num_workers 8 --server arc_fastscratch --wandb

# Change slurm name
# Change model name
# output_path
# consider changing hyperparams
# name
# Ensure wandb
# Ensure previous command is commented out