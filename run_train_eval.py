import argparse
import time
from pathlib import Path

import main_simsiam
import main_lincls
import models
import penalties
import utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}

parser = argparse.ArgumentParser(description='SimSiam Full Pipeline')
# ======================================================================================================================
# ===============================================   PRETRAIN PARAMETER   ===============================================
# ======================================================================================================================
# Model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2048), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0.0005)',
                    dest='weight_decay')

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')

# Misc
parser.add_argument("--dataset", default="ImageNet", type=str, choices=["ImageNet", "CIFAR10", "CIFAR100"],
                    help="Specify the name of your dataset. Choose from: ImageNet, CIFAR10, CIFAR100")
parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                    help='Specify path to the training data.')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument("--resize_all_inputs", default=False, type=utils.bool_flag,
                    help="Resizes all images of the ImageNet dataset to one size. Here: 224x224")
parser.add_argument("--resize_input", default=False, type=utils.bool_flag,
                    help="Set this flag to resize the images of the dataset, images will be resized to the value given "
                         "in parameter --resize_size (default: 512). Can be useful for datasets with varying resolutions.")
parser.add_argument("--resize_size", default=512, type=int,
                    help="If resize_input is True, this will be the maximum for the longer edge of the resized image.")

# STN parameters
parser.add_argument('--stn_mode', default='affine', type=str,
                    help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                         'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
parser.add_argument('--stn_pretrained_weights', default='', type=str,
                    help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
parser.add_argument("--invert_stn_gradients", default=False, type=utils.bool_flag,
                    help="Set this flag to invert the gradients used to learn the STN")
parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate optimizer for the STN parameters; "
                         "annealed with cosine and no warmup")
parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
    linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate localization network for each head.")
parser.add_argument("--summary_writer_freq", default=1e6, type=int,
                    help="Defines the number of iterations the summary writer will write output.")
parser.add_argument("--grad_check_freq", default=1e6, type=int,
                    help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
parser.add_argument("--stn_res", default=(224, 96), type=int, nargs='+',
                    help="Set the resolution of the global and local crops of the STN (default: 224x and 96x)")
parser.add_argument("--use_unbounded_stn", default=False, type=utils.bool_flag,
                    help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                    help="Specifies the number of warmup epochs for the STN (default: 0).")
parser.add_argument("--stn_conv1_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
parser.add_argument("--stn_conv2_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv2 for the STN localization network (default: 32).")
parser.add_argument("--stn_theta_norm", default=False, type=utils.bool_flag,
                    help="Set this flag to normalize 'theta' in the STN before passing to affine_grid(theta, ...). Fixes the problem with cropping of the images (black regions)")
parser.add_argument("--use_stn_penalty", default=False, type=utils.bool_flag,
                    help="Set this flag to add a penalty term to the loss. Similarity between input and output image of STN.")
parser.add_argument("--penalty_loss", default="ThetaLoss", type=str, choices=penalty_list,
                    help="Specify the name of the similarity to use.")
parser.add_argument("--epsilon", default=1., type=float,
                    help="Scalar for the penalty loss. Rescales the gradient by multiplication.")
parser.add_argument("--invert_penalty", default=False, type=utils.bool_flag,
                    help="Invert the penalty loss.")
parser.add_argument("--stn_color_augment", default=False, type=utils.bool_flag, help="todo")
parser.add_argument("--summary_plot_size", default=16, type=int,
                    help="Defines the number of samples to show in the summary writer.")
# ======================================================================================================================
# =================================================   EVAL PARAMETER   =================================================
# ======================================================================================================================
# Model parameters
parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--img_size', default=32, type=int, help='the resolution of the images')
# ======================================================================================================================
# ===============================================   PIPELINE PARAMETER   ===============================================
# ======================================================================================================================
parser.add_argument("--pipeline_mode", default=('pretrain', 'eval'), choices=['pretrain', 'eval'], type=str, nargs='+')


if __name__ == "__main__":
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'pretrain' in args.pipeline_mode:
        print('STARTING PRETRAINING')
        main_simsiam.train_simsiam(args)
        time.sleep(10)
        print('FINISHED PRETRAINING')

    if 'eval' in args.pipeline_mode:
        # change linear specific parameters
        args.epochs = 200
        args.lr = 0.01
        args.momentum = 0.9
        args.weight_decay = 0
        args.batch_size = 768
        args.pretrained = f"{args.output_dir}/checkpoint.pth"
        print('STARTING EVALUATION')
        main_lincls.eval_linear(args, True)
        print('FINISHED EVALUATION')
