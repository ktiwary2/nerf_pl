#!/bin/bash
#SBATCH -n 20 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python3 train_variable_light.py --img_wh 64 64 --batch_size 4096 --shadow_mapper2d\
 --exp_name BUNNY_SIGMA100_64x64_sm2_smapper2d_nimp128_nsamp64 --use_disp --shadow_method shadow_method_2\
 --num_epochs 500 --root_dir ../../datasets/variable_light/results_200_bunny_var_light_fov50_sigma100/ --N_importance 128\
 --N_samples 64 --ckpt_path ./ckpts/BUNNY_SIGMA100_64x64_sm2_smapper2d_nimp128_nsamp64/epoch=5.ckpt