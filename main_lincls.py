# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import models
import utils

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def eval_linear(args, dist_inited=False):
    utils.init_distributed_mode(args) if not dist_inited else None
    cudnn.benchmark = True

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    with (Path(args.output_dir) / "settings.eval").open("w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # infer per gpu batch size
    args.batch_size_per_gpu = int(args.batch_size / args.world_size)
    # ============ preparing data ... ============
    dataset_val, args.num_labels = build_dataset(is_train=False, args=args)
    sampler = torch.utils.data.SequentialSampler(dataset_val)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args.num_labels, True)
    model.cuda()
    model.eval()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))
        sys.exit(1)

    criterion = nn.CrossEntropyLoss().cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    dataset_train, args.num_labels = build_dataset(is_train=True, args=args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    writer = None
    if utils.is_main_process() == 0:
        path = Path(args.output_dir).joinpath("summary")
        writer = SummaryWriter(path)

    print("Setup completed ---> Starting Training and Evaluation")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, init_lr, epoch, args)

        train_stats = train(train_loader, model, criterion, optimizer, epoch, writer)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model)
            print(
                f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            if writer:
                writer.add_scalar(tag="acc1", scalar_value=test_stats["acc1"], global_step=epoch)
                writer.add_scalar(tag="acc5", scalar_value=test_stats["acc5"], global_step=epoch)
                writer.add_scalar(tag="best-acc", scalar_value=best_acc, global_step=epoch)

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.eval").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(loader, model, criterion, optimizer, epoch, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for it, (samples, targets) in enumerate(metric_logger.log_every(loader, 20, header)):
        # global iteration
        it = len(loader) * epoch + it
        # move to gpu
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        output = model(samples)
        loss = criterion(output, targets)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer:
            writer.add_scalar(tag="loss(eval)", scalar_value=loss.item(), global_step=it)
            writer.add_scalar(tag="lr(eval)", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(loader, model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=" ")
    header = 'Test:'
    for samples, targets in metric_logger.log_every(loader, 20, header):
        # move to gpu
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # forward
        output = model(samples)
        loss = nn.CrossEntropyLoss()(output, targets)

        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset == 'CIFAR10':
        return datasets.CIFAR10(args.data_path, download=True, train=is_train, transform=transform), 10
    if args.dataset == 'CIFAR100':
        return datasets.CIFAR100(args.data_path, download=True, train=is_train, transform=transform), 100
    elif args.dataset == 'ImageNet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        return dataset, 1000
    print(f"Does not support dataset: {args.dataset}")
    sys.exit(1)


def build_transform(is_train, args):
    if args.dataset == 'CIFAR10':
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        factor = args.img_size // 32
        return transforms.Compose([
            transforms.Resize(args.img_size + factor * 4, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            normalize,
        ])
    if args.dataset == 'ImageNet':
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            normalize,
        ])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--batch_size', default=768, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--img_size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")

    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    # Dataset parameters
    parser.add_argument('--dataset', default="ImageNet", choices=["ImageNet", "CIFAR10", "CIFAR100"], type=str,
                        help='Specify name of dataset (default: ImageNet)')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str, help='Specify path to your dataset.')
    # distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str, help='url used to set up distributed training')
    # Misc
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_linear(args)
