#!/bin/bash
#SBATCH -n 20 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_efficient_sm.py --dataset_name efficient_sm\
 --root_dir /home/gridsan/ktiwary/datasets/volumetric/results_500_v2_distance_transform_150/ --N_importance 64\
 --N_samples 64 --num_gpus 0 1 --img_wh 128 128 --noise_std 0 --num_epochs 300 --optimizer adam --lr 0.00001\
 --exp_name RAY_TERM_ONLY_128x128_single_image_grad_on_sm2_sigma_run3_2gpu --num_sanity_val_steps 1 --Light_N_importance 64\
 --grad_on_light --batch_size 512
 