#!/bin/bash

module reset
module load Anaconda3/2020.11
module load GCC/11.2.0
conda init 
source ~/.bashrc
conda activate py39
which python

echo "Starting eval..."

# # Resnet 18
# python basic_evaluate.py --model_name resnet18 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_r18_wbce_basic/ckpt_9229_S9229_tid_r18_basic_fishair_processed_resnet18.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name resnet18_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "Resnet18 failed."
#     exit 1
# fi

# # Resnet 34
# python basic_evaluate.py --model_name resnet34 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_r34_wbce_basic/ckpt_9229_S9229_r34_wbce_basic_fishair_processed_resnet34.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name resnet34_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "Resnet34 failed."
#     exit 1
# fi

# # swinb
# python basic_evaluate.py --model_name swin_b --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_swinb_wbce_basic/ckpt_9229_S9229_tid_swinb_basic_fishair_processed_swin_b.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate/ --name swinb_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "swin_b failed."
#     exit 1
# fi

# # vgg19
# python basic_evaluate.py --model_name vgg19 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vgg19_wbce_basic/ckpt_9229_S9229_tid_vgg19_basic_fishair_processed_vgg19.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate/ --name vgg19_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "vgg19 failed."
#     exit 1
# fi

# # vit16
# python basic_evaluate.py --model_name vit_b_16 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vit16_wbce_basic/ckpt_9229_S9229_tid_vit16_basic_fishair_processed_vit_b_16.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name vit_b_16_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "vit_b_16 failed."
#     exit 1
# fi

# # vit32
# python basic_evaluate.py --model_name vit_b_32 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_vit32_wbce_basic/ckpt_9229_S9229_tid_vit32_basic_fishair_processed_vit_b_32.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name vit_b_32_corrected --server arc --batch_size 256 --num_workers 8

# if [ $? -ne 0 ]; then
#     echo "vit_b_32 failed."
#     exit 1
# fi

# Inception-v3
python basic_evaluate.py --model_name inception_v3 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/inception_v3_wbce_basic/ckpt_9229_S9229_tid_iv3_basic_fishair_processed_inception_v3.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name inception_v3_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "Inception v3 failed."
    exit 1
fi

# convnext_base
python basic_evaluate.py --model_name convnext_base --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/convnext_base_wbce_basic/ckpt_9229_S9229_tid_convnext_base_basic_fishair_processed_convnext_base.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name convnext_base_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "convnext_base failed."
    exit 1
fi

# efficientnet_v2_m
python basic_evaluate.py --model_name efficientnet_v2_m --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/efficientnet_v2_m_wbce_basic/ckpt_9229_S9229_tid_efficientnet_v2_m_basic_fishair_processed_efficientnet_v2_m.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name efficientnet_v2_m_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "efficientnet_v2_m failed."
    exit 1
fi

# mobilenet_v3_large

python basic_evaluate.py --model_name mobilenet_v3_large --checkpoint_path  /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/mobilenet_v3_large_wbce_basic/ckpt_9229_S9229_mobilenet_v3_large_fishair_processed_mobilenet_v3_large.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name mobilenet_v3_large_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "mobilenet_v3_large failed."
    exit 1
fi

# maxvit_t
python basic_evaluate.py --model_name maxvit_t --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/maxvit_t_wbce_basic/ckpt_9229_S9229_tid_maxvit_t_basic_fishair_processed_maxvit_t.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name maxvit_t_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "maxvit_t failed."
    exit 1
fi

# resnext50_32x4d
python basic_evaluate.py --model_name resnext50_32x4d --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/resnext50_32x4d_wbce_basic/ckpt_9229_S9229_resnext50_32x4d_fishair_processed_resnext50_32x4d.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name resnext50_32x4d_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "resnext50_32x4d failed."
    exit 1
fi

# cvt_13
python basic_evaluate.py --model_name cvt_13 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/ResumedRuns/cvt_13_wbce_basic/ckpt_9229_S9229_tid_cvt_13_basic_fishair_processed_cvt_13.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name cvt_13_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "cvt_13 failed."
    exit 1
fi

# regnet_y
python basic_evaluate.py --model_name regnet_y --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_regnet_y_wbce_basic/ckpt_9229_S9229_regnet_y_wbce_basic_fishair_processed_regnet_y.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name regnet_y_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "regnet_y failed."
    exit 1
fi

# pvt_v2
python basic_evaluate.py --model_name pvt_v2 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_pvt_v2_wbce_basic/ckpt_9229_S9229_pvt_v2_wbce_basic_fishair_processed_pvt_v2.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name pvt_v2_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "pvt_v2 failed."
    exit 1
fi

# swinb_22k
python basic_evaluate.py --model_name swinb_22k --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_swinb_22k_wbce_basic/ckpt_9229_S9229_swinb_22k_wbce_basic_fishair_processed_swinb_22k.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name swinb_22k_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "swinb_22k failed."
    exit 1
fi

# mobile_vit_xs
python basic_evaluate.py --model_name mobile_vit_xs --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_mobile_vit_xs_wbce_basic/ckpt_9229_S9229_mobile_vit_xs_wbce_basic_fishair_processed_mobile_vit_xs.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name mobile_vit_xs_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "mobile_vit_xs failed."
    exit 1
fi

# mobile_vit_v2
python basic_evaluate.py --model_name mobile_vit_v2 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_mobile_mobile_vit_v2_wbce_basic/ckpt_9229_S9229_mobile_vit_v2_wbce_basic_fishair_processed_mobile_vit_v2.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name mobile_vit_v2_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "mobile_vit_v2 failed."
    exit 1
fi

# diet_distilled_s
python basic_evaluate.py --model_name diet_distilled_s --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_diet_distilled_s_wbce_basic/ckpt_9229_S9229_diet_distilled_s_wbce_basic_fishair_processed_diet_distilled_s.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/CorrectedEvaluate --name diet_distilled_s_corrected --server arc --batch_size 256 --num_workers 8

if [ $? -ne 0 ]; then
    echo "diet_distilled_s failed."
    exit 1
fi

# Change model name
# Change ckpt path
# Change output path
# Change name
# Change echo


echo "All scripts ran successfully."