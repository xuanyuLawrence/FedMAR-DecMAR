import argparse
import datetime
from inspect import trace
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import model_mae

from engine_pretrain import train_one_epoch, train_contra_CNN_one_epoch, train_MAE_CNN_one_epoch, train_contra_ViT_one_epoch, evaluate
from util.pos_embed import interpolate_pos_embed
from federated_learning import client_resume_from_federated

from debug import ckpt_diff
import copy
import baseline_models as bm
from model_mae_resnet import ResNetMaskedAutoencoder
import model_ViT_simsiam

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import ammd


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    ##
    parser.add_argument('--model', default='mae_vit_tiny_p16_nodp', type=str, metavar='MODEL',
                        help='Name of model to train')
    # parser.add_argument('--model', default='mae_vit_tiny_p16_nodp', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    ##
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--save_interval', default=10, type=int, metavar='N',
                        help='interval to save a model')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    ##
    parser.add_argument('--output_dir', default='./checkpoint/client',
                        help='path where to save, empty for no saving')
    ##                    
    parser.add_argument('--log_dir', default='./checkpoint/client',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def pretrain_init(main_args, client_ID, dataset_train):
    args = get_args_parser()
    args = args.parse_args()

    args.batch_size = torch.cuda.device_count() * args.batch_size

    if client_ID:
        args.output_dir = '%sclient/%s' % (main_args.save_path, client_ID)
        args.log_dir = '%sclient/%s' % (main_args.save_path, client_ID)
    else:
        args.output_dir = '%sserver/' % main_args.save_path
        args.log_dir = '%sserver/' % main_args.save_path

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    torch.manual_seed(main_args.seed)
    np.random.seed(main_args.seed)

    cudnn.benchmark = True

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if dataset_train:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
    else:
        if dataset_train:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = None
    if dataset_train:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        # args.lr = args.blr

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    return args, data_loader_train, device, log_writer


def pretrain_MAE(main_args, client_ID, dataset_train, depths, r_eps, 
                global_latent=None, gamma=None, mask_noise=None, client_local_latent=None, load_path=None, save_path=None):

    args, data_loader_train, device, log_writer = pretrain_init(main_args, client_ID, dataset_train)

    print("Mask ratio: %s" % args.mask_ratio)
    
    # define the model
    model = model_mae.__dict__[args.model](depth=depths[0], decoder_depth=depths[1], norm_pix_loss=args.norm_pix_loss)
    
    if load_path and os.path.exists(load_path):
        ckpt_fed = torch.load(load_path, map_location='cpu')
        model.load_state_dict(ckpt_fed['model'], strict=False)
        print("Load checkpoint from: %s" % load_path)     

    model.to(device)
    if isinstance(mask_noise, torch.Tensor):
        model.set_align_mask_noise(mask_noise.to(device))
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # Compute Gflops
    # model.eval()
    # input_tensor = torch.randn(1, 3, 224, 224).to('cuda')  # e.g., for ViT or ResNet

    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    # raise TypeError


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    latents = []
    local_latent = None
    prev_local_latent = None
    if client_local_latent:
        if isinstance(client_local_latent[client_ID], torch.Tensor):
            prev_local_latent = client_local_latent[client_ID]
    print("with prev_local_latent: %s" % prev_local_latent)


    if gamma:
        c_gamma = gamma[client_ID]['gamma']
    else:
        c_gamma = 0

    if dataset_train:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        best_loss = float('inf')

        latents = []

        for epoch in range(args.start_epoch, args.start_epoch + r_eps):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)   
            train_stats, avg_latent = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                global_latent=global_latent,
                gamma=c_gamma,
                prev_latent=prev_local_latent,
                args=args
            )
            
            if isinstance(avg_latent, torch.Tensor):
                latents.append(avg_latent)
            # if len(local_latents) > 0:
            #     for latent in local_latents:
            #         latents.append(latent)

            if args.output_dir:
                if train_stats['loss'] < best_loss:
                    best_loss = train_stats['loss']
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, best=True)
                if (epoch + 1) % args.save_interval == 0:
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")  

        if len(latents) > 0:
            local_latent = sum(latents) / len(latents)
            # local_latent = latents[-1]
        # if len(local_latents) > 0:
            # local_latent = ammd.geometric_median_tensor(latents)
            if client_local_latent:
                client_local_latent[client_ID] = local_latent

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    model.to("cpu")
    if isinstance(mask_noise, torch.Tensor):
        mask_noise.cpu()
        del mask_noise
    torch.cuda.empty_cache()


    new_checkpoint = {
        'model': model_without_ddp.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'loss_scaler': loss_scaler.state_dict(),
        #'lr': train_stats['lr']
    }

    if dataset_train:
        new_checkpoint['loss'] = train_stats['loss']
    
    if save_path:
        misc.save_on_master(new_checkpoint, save_path)
    
    return new_checkpoint, local_latent

def pretrain_contra_CNN(main_args, client_ID, dataset_train, depth, r_eps, 
                load_path=None, save_path=None):

    args, data_loader_train, device, log_writer = pretrain_init(main_args, client_ID, dataset_train)
    
    # define the model
    model = bm.simsiam('res%s-origin' % depth)
    # model = bm.create_backbone('res%s-origin' % depth, num_classes=num_classes)

    if load_path and os.path.exists(load_path):
        ckpt_fed = torch.load(load_path, map_location='cpu')
        model.load_state_dict(ckpt_fed['model'], strict=False)
        print("Load checkpoint from: %s" % load_path)    

    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module


    optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # print(optimizer)
    loss_scaler = NativeScaler()

    if dataset_train:
        print(f"Start training for {r_eps} epochs")
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(r_eps):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)  

            train_stats = train_contra_CNN_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

            if args.output_dir:
                if train_stats['loss'] < best_loss:
                    best_loss = train_stats['loss']
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, best=True)
                if epoch % args.save_interval == 0:
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")  

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    model.to("cpu")
    torch.cuda.empty_cache()

    new_checkpoint = {
        'model': model_without_ddp.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'loss_scaler': loss_scaler.state_dict(),
        #'lr': train_stats['lr']
    }

    if dataset_train:
        new_checkpoint['loss'] = train_stats['loss']

    if save_path:
        misc.save_on_master(new_checkpoint, save_path)

    return new_checkpoint

def pretrain_MAE_CNN(main_args, client_ID, dataset_train, depth, r_eps, 
                load_path=None, save_path=None):

    args, data_loader_train, device, log_writer = pretrain_init(main_args, client_ID, dataset_train)
    
    # define the model
    model = ResNetMaskedAutoencoder()

    if load_path and os.path.exists(load_path):
        ckpt_fed = torch.load(load_path, map_location='cpu')
        model.load_state_dict(ckpt_fed['model'], strict=False)
        print("Load checkpoint from: %s" % load_path)    

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module


    optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # print(optimizer)
    loss_scaler = NativeScaler()

    if dataset_train:
        print(f"Start training for {r_eps} epochs")
        start_time = time.time()
        best_loss = float('inf')
        
        for epoch in range(r_eps):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)  

            train_stats = train_MAE_CNN_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

            if args.output_dir:
                if train_stats['loss'] < best_loss:
                    best_loss = train_stats['loss']
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, best=True)
                if epoch % args.save_interval == 0:
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")  

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    model.to("cpu")
    torch.cuda.empty_cache()

    new_checkpoint = {
        'model': model_without_ddp.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'loss_scaler': loss_scaler.state_dict(),
        #'lr': train_stats['lr']
    }

    if dataset_train:
        new_checkpoint['loss'] = train_stats['loss']

    if save_path:
        misc.save_on_master(new_checkpoint, save_path)

    return new_checkpoint

def pretrain_contra_ViT(main_args, client_ID, dataset_train, depths, r_eps, 
                load_path=None, save_path=None):

    args, data_loader_train, device, log_writer = pretrain_init(main_args, client_ID, dataset_train)
    
    # define the model
    model = model_ViT_simsiam.__dict__[args.model](depth=depths[0],norm_pix_loss=args.norm_pix_loss)
    
    if load_path and os.path.exists(load_path):
        ckpt_fed = torch.load(load_path, map_location='cpu')
        model.load_state_dict(ckpt_fed['model'], strict=False)
        print("Load checkpoint from: %s" % load_path)     

    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    if dataset_train:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        best_loss = float('inf')

        for epoch in range(args.start_epoch, args.start_epoch + r_eps):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)   
            train_stats = train_contra_ViT_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

            if args.output_dir:
                if train_stats['loss'] < best_loss:
                    best_loss = train_stats['loss']
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, best=True)
                if (epoch + 1) % args.save_interval == 0:
                    misc.save_model_pretrain(
                        args=args, epoch=epoch, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")  

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
    model.to("cpu")
    torch.cuda.empty_cache()

    new_checkpoint = {
        'model': model_without_ddp.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'loss_scaler': loss_scaler.state_dict(),
        #'lr': train_stats['lr']
    }

    if dataset_train:
        new_checkpoint['loss'] = train_stats['loss']
    
    if save_path:
        misc.save_on_master(new_checkpoint, save_path)
    
    return new_checkpoint
