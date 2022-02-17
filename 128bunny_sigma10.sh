#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:2

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerfpl_dev

# Run the script
python3 train_variable_light.py --img_wh 128 128 --exp_name BUNNY_SIGMA10_128x128_sm2_smapper2d_run2_random_init --use_disp\
 --shadow_method shadow_method_2 --num_epochs 500\
 --root_dir ../../datasets/variable_light/results_200_bunny_var_light_fov50_sigma10/ --N_importance 32\
  --N_samples 64 --num_gpus 2 --batch_size 256 --optimizer adam --lr 0.00001