import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--haz_dir_name', type=str)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')

    # loss weight
    parser.add_argument('--composite_weight', type=float, default=1)
    parser.add_argument('--distortion_weight', type=float, default=1e-3)
    parser.add_argument('--opacity_weight', type=float, default=1e-3)
    parser.add_argument('--dcp_weight', type=float, default=6e-3)
    parser.add_argument('--foggy_weight', type=float, default=6e-5)

    # validation options
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load)')

    return parser.parse_args()
