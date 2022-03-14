#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerfpl-v2

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm\
 --root_dir /home/gridsan/ktiwary/datasets/variable_cam/results_500_v2_sphere_sigma50/\
 --N_importance 128 --N_samples 64 --num_gpus 0 --img_wh 64 64 --noise_std 0 --num_epochs 500\
 --optimizer adam --lr 0.00001 --exp_name UPDATED_LT_SPHERE_SIGMA_50_64x64_run1_e8_cont\
 --num_sanity_val_steps 1\
 --Light_N_importance 128 --grad_on_light --batch_size 4096\
 --ckpt_path ./eff_sm_updated_light_matrix_NEW_mar02/ckpts/UPDATED_LT_SPHERE_SIGMA_50_64x64_run1_e8/epoch=295.ckpt


# 14385831

# sphere:14385832