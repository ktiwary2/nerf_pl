#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --N_importance 128 --N_samples 64 --num_gpus 0 1 --img_wh 128 128 --chunk 32768 --noise_std 0 --num_epochs 200 --batch_size 1024 --optimizer adam --lr 0.00001 --exp_name Nimp_128_128x128_grad_off_light_LN_16_shadowmethod1_blur_run3 --Light_N_importance 32 --shadow_method shadow_method_1 --blur 2 > ./logs/Nimp_128_128x128_grad_off_light_LN_16_shadowmethod1_blur_run3.log
