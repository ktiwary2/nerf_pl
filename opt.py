import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', "pyredner", 'shadows', 'efficient_sm', 'pyredner2'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[128, 128],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*8,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', nargs="+", type=int, default=[0, 1],
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    parser.add_argument('--black_and_white_test', default=False, action="store_true",
                        help='Converts all images to black and white images.')

    parser.add_argument('--num_sanity_val_steps', default=1, type=int,
                        help='Converts all images to black and white images.')

    parser.add_argument('--Light_N_importance', type=int, default=0,
                        help='number of additional fine samples')
                        
    parser.add_argument('--sample_light_depth_every', default=5, type=int,
                        help='sample_light_depth_every batch of rays.')

    parser.add_argument('--grad_on_light', default=False, action="store_true",
                        help='Calculate gradient on Light. If you set this to true, we will sample light depth every batch!')

    parser.add_argument('--shadow_method', type=str, default='shadow_method_1',
                        choices=['shadow_method_1', 'shadow_method_2'],
                        help='shadow method to use')

    parser.add_argument('--coords_trans', default=False, action="store_true",
                        help='Perform coords_trans on the blender coords to pyt them in the standard coordinate system.')

    parser.add_argument('--coords_trans2', default=False, action="store_true",
                        help='Perform coords_trans on the blender coords to pyt them in the standard coordinate system.')

    parser.add_argument('--blur', default=False, action="store_true",
                        help='blur image with radius 5.')


    return parser.parse_args()
