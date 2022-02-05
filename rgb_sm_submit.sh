#!/bin/bash
#SBATCH -n 40 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_rgb_sm_juntos.py --dataset_name rgb_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/\
 --N_importance 128 --N_samples 64 --num_gpus 0 --img_wh 128 128 --noise_std 0 --num_epochs 300 --batch_size 512 --optimizer adam\
 --lr 0.00001 --exp_name sigma/sigma_light_inside_Nimp128_128x128_grad_on_LN32_sm2_blur5 --num_sanity_val_steps 1 --grad_on_light\
 --Light_N_importance 32 --shadow_method shadow_method_2 --blur 5 > ./sigma_out/sigma_light_inside_Nimp128_128x128_grad_on_LN32_sm2_blur5.out