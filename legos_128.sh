#!/bin/bash
#SBATCH -n 20 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_rgb_sm_juntos.py --dataset_name rgb_sm --root_dir ../../datasets/volumetric/results_500_lego_shadows/\
 --N_importance 64 --N_samples 64 --num_gpus 0 --img_wh 128 128 --noise_std 0 --num_epochs 200 --batch_size 512\
 --optimizer adam --lr 0.00001 --exp_name legos/128x128_Nimp64_LN32_grad_on_bluroff --Light_N_importance 32 --shadow_method shadow_method_2\
 --grad_on_light > ./sigma_out/legos_128x128_Nimp64_LN32_grad_on_bluroff.out