#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --N_importance 128 --N_samples 64 --num_gpus 0 --img_wh 64 64 --noise_std 0 --num_epochs 200 --batch_size 4096 --optimizer adam --lr 0.00001 --exp_name Nimp_128_64x64_grad_on_light_LN_64_shadowmethod1_blur_run3 --Light_N_importance 64 --shadow_method shadow_method_1 --blur 2 > ./logs/Nimp_128_64x64_grad_on_light_LN_64_shadowmethod1_blur_run3.log
