#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm\
 --root_dir /home/gridsan/ktiwary/datasets/variable_cam_v2/results_empire_state_v1_sigma150/\
 --N_importance 128 --N_samples 64\
 --num_gpus 0 --img_wh 64 64 --num_epochs 300\
 --exp_name EMPIRE_sigma150_var_cam_run2 --num_sanity_val_steps 1\
 --Light_N_importance 128 --grad_on_light --batch_size 4096