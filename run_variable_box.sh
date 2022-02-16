#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerfpl_dev

# Run the script
python3 train_variable_light.py --img_wh 64 64 --batch_size 4096 --shadow_mapper2d --exp_name opacity_loss_64x64_nimp128_box_fov50_nosigma_randominit --use_disp --shadow_method shadow_method_2 --num_epochs 500 --root_dir ../../datasets/variable_light/results_200_box_fov50/ --N_importance 128 --N_samples 64