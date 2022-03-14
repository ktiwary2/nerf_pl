#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm\
 --root_dir ../../datasets/variable_cam/results_500_v2_bunny_var_cam_sigma30/\
 --N_importance 128 --N_samples 128\
 --num_gpus 0 --img_wh 64 64 --noise_std 0 --num_epochs 300 --optimizer adam --lr 0.00001\
 --exp_name UPDATED_LT_BUNNY_SIGMA150_to50_run1 --num_sanity_val_steps 1\
 --Light_N_importance 128 --grad_on_light --batch_size 4096\ 
 --ckpt_path ./UPDATED_LT_BUNNY_SIGMA150_scratch_run1_e195.ckpt
