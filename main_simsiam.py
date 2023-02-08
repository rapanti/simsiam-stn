import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import models

import penalties
import simsiam.loader
import simsiam.builder
from stn import AugmentationNetwork, STN
import utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}


def train_simsiam(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "settings.json").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = simsiam.builder.SimSiam(
        models.__dict__[args.arch],
        args.dim, args.pred_dim)

    model = model.cuda()

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    print(model)  # print model after SyncBatchNorm

    print("=> creating STN")
    _stn = STN(
        mode=args.stn_mode,
        invert_gradients=args.invert_stn_gradients,
        separate_localization_net=args.separate_localization_net,
        conv1_depth=args.stn_conv1_depth,
        conv2_depth=args.stn_conv2_depth,
        theta_norm=args.stn_theta_norm,
        global_crops_number=2,
        local_crops_number=0,
        global_crops_scale=(0.2, 1),
        resolution=args.stn_res,
        unbounded_stn=args.use_unbounded_stn,
    )

    stn = AugmentationNetwork(
        transform_net=_stn,
        resize_input=args.resize_input,
        resize_size=args.resize_size,
    )

    stn = stn.cuda()

    if utils.has_batchnorms(stn):
        stn = nn.SyncBatchNorm.convert_sync_batchnorm(stn)

    stn = torch.nn.parallel.DistributedDataParallel(stn, device_ids=[args.gpu])
    print(stn)  # print model after SyncBatchNorm

    criterion = nn.CosineSimilarity(dim=1).cuda()

    stn_penalty = penalty_dict[args.penalty_loss](
        invert=args.invert_penalty,
        eps=args.epsilon,
        global_crops_scale=(0.2, 1),
    ).cuda() if args.use_stn_penalty else None

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ============ preparing data ... ============
    dataset = utils.build_dataset(True, args)
    sampler = DistributedSampler(dataset, shuffle=True)
    args.batch_size_per_gpu = int(args.batch_size / utils.get_world_size())
    train_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ ColorAugments after STN ============
    color_augment = utils.ColorAugmentation() if args.stn_color_augment else None

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        stn=stn,
    )
    start_epoch = to_restore["epoch"]

    summary_writer = None
    if utils.is_main_process():
        summary_writer = utils.SummaryWriterCustom(log_dir=Path(args.output_dir) / "summary",
                                                   plot_size=args.summary_plot_size)

    start_time = time.time()
    print("Starting SimSiam training !")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, init_lr, epoch, args)

        # ============ training one epoch of SimSiam ============
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args,
                            stn, stn_penalty, color_augment, summary_writer)
        # ============ writing logs ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
            'stn': stn.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(train_loader, model, criterion, optimizer, epoch, args,
          stn, stn_penalty, color_augment, summary_writer):
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    # switch to train mode
    model.train()
    for it, (images, _) in enumerate(metric_logger.log_every(train_loader, 10, header)):
        it = len(train_loader) * epoch + it  # global training iteration

        # move images to gpu
        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
        else:
            images = images.cuda(non_blocking=True)

        stn_images, thetas = stn(images)

        penalty = torch.tensor(0.).cuda()
        if stn_penalty:
            penalty = stn_penalty(images=stn_images, target=images, thetas=thetas)
            # overlap = overlap_penalty(thetas)

        if color_augment:
            images = color_augment(stn_images)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        siam = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        loss = penalty + siam

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        if stn_penalty:
            metric_logger.update(siam=siam.item())
            metric_logger.update(penalty=penalty.item())
            # metric_logger.update(overlap=overlap.item())

        if utils.is_main_process():
            summary_writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=it)
            summary_writer.add_scalar(tag="lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)
            summary_writer.add_scalar(tag="weight decay", scalar_value=optimizer.param_groups[0]["weight_decay"],
                                      global_step=it)
            if args.use_stn_penalty:
                summary_writer.add_scalar(tag="siam", scalar_value=siam.item(), global_step=it)
                summary_writer.add_scalar(tag="penalty", scalar_value=penalty.item(), global_step=it)
                # summary_writer.add_scalar(tag="overlap", scalar_value=overlap.item(), global_step=it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('SimSiam', add_help=False)

    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size (default: 512), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

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

    # simsiam specific configs:
    parser.add_argument('--dim', default=2048, type=int,
                        help='feature dimension (default: 2048)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')
    parser.add_argument('--fix-pred-lr', action='store_true',
                        help='Fix learning rate for the predictor')

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

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_simsiam(args)
