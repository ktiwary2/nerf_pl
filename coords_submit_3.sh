#!/bin/bash
#SBATCH -n 8 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --N_importance 64 --N_samples 64 --num_gpus 0 1 --img_wh 128 128 --noise_std 0 --num_epochs 300 --batch_size 1024 --optimizer adam --lr 0.00001 --exp_name COORDS_TRANS_Nimp_64_128x128_grad_on_light_LN_8_shadowmethod2_blur_run3 --num_sanity_val_steps 0 --Light_N_importance 8 --shadow_method shadow_method_2 --blur --coords_trans --grad_on_light > ./logs/COORDS_TRANS_Nimp_64_128x128_grad_on_light_LN_8_shadowmethod2_blur_run3.log