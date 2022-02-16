#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerfpl_dev

# Run the script
python3 train.py --root_dir ../../datasets/volumetric/results_500_light_inside_bounding_vol_v1/ --exp run2_BASELINE-light_inside_bounding_128x128\
 --img_wh 128 128 --num_gpus 1 --batch_size 8192 --dataset_name blender