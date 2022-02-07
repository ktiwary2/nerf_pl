#!/bin/bash
#SBATCH -n 40 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python train_rgb_sm_juntos.py --dataset_name rgb_sm --root_dir ../../datasets/volumetric/results_500_lego_shadows/\
 --N_importance 64 --N_samples 64 --num_gpus 0 1 --img_wh 200 200 --noise_std 0 --num_epochs 300 --batch_size 512\
 --optimizer adam --lr 0.00001 --exp_name legos/200x200_Nimp64_LN32_sm2_grad_off_bluroff --Light_N_importance 32\
 --shadow_method shadow_method_2 --sm_weight 1.0 --rgb_weight 2.0 > ./legos/200x200_Nimp64_LN32_sm2_grad_off_bluroff.out