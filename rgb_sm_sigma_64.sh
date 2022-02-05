#!/bin/bash
#SBATCH -n 40 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_rgb_sm_juntos.py --dataset_name rgb_sm --root_dir ../../datasets/volumetric/results_500_v2_distance_transform_150/\
 --N_importance 64 --N_samples 64 --num_gpus 0 --img_wh 64 64 --noise_std 0 --num_epochs 200 --batch_size 4096 --optimizer adam\
 --lr 0.00001 --exp_name sigma/sigma_150_Nimp_64_64x64_grad_on_light_LN_32_sm2 --num_sanity_val_steps 1 --Light_N_importance 32\
 --shadow_method shadow_method_2 --grad_on_light >  ./sigma_out/sigma_150_Nimp_64_64x64_grad_on_light_LN_32_sm2.out