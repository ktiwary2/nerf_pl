#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerfpl_dev

# Run the script
python3 train_variable_light.py --img_wh 128 128 --exp_name 128x128_sm2_smapper2d_nimp128_nsamp64_sigma\
 --use_disp --shadow_method shadow_method_2 --num_epochs 500\
 --root_dir ../../datasets/variable_light/results_500_chair_sigma_150/ --N_importance 32\
 --N_samples 64 --num_gpus 2 --batch_size 256