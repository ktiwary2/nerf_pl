#!/bin/bash
#SBATCH -n 20 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_rgb_sm_juntos.py --dataset_name rgb_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --N_importance 128 --N_samples 64 --num_gpus 0 1 --img_wh 200 200 --noise_std 0 --num_epochs 200 --batch_size 128 --optimizer adam --lr 0.00001 --exp_name Nimp_128_200x200_grad_on_light_LN_32_shadowmethod2_blur_10_1 --num_sanity_val_steps 1 --grad_on_light --Light_N_importance 32 --shadow_method shadow_method_2 --blur 10