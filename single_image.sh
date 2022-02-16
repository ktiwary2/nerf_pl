#!/bin/bash
#SBATCH -n 20 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm --root_dir /home/gridsan/ktiwary/datasets/volumetric/results_500_v2/\
 --N_importance 64 --N_samples 64 --num_gpus 0 --img_wh 64 64 --noise_std 0 --num_epochs 300 --batch_size 4096\
 --optimizer adam --lr 0.00001 --exp_name single_image_grad_on_sm2_no_sigma --num_sanity_val_steps 1 --Light_N_importance 64\
 --shadow_method shadow_method_2