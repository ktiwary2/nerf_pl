#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --N_importance 64 --N_samples 64 --num_gpus 0 --img_wh 32 32 --noise_std 0 --num_epochs 300 --batch_size 4096 --optimizer adam --lr 0.00001 --exp_name Nimp_64_32x32_grad_on_light_LN_0_shadowmethod2_run1 --grad_on_light --num_sanity_val_steps 1 --Light_N_importance 0 --shadow_method shadow_method_2 > ./logs/Nimp_64_32x32_grad_on_light_LN_0_shadowmethod2_run1.log