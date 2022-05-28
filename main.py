# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# For Testing put the configuration as follows:
# 1) coco
# --batch_size 2 --no_aux_loss --eval --resume weights/detr-r50-e632da11.pth --coco_path /media/user/x_2/coco2017
# 2) cityscapes
# --batch_size 2 --no_aux_loss --eval --resume weights/detr-r50-e632da11.pth
# --coco_path /media/user/data/eslam/CityScapes_Dataset/leftImg8bit_trainvaltest --dataset_file "cityscapes"
# 3) Kitti-Tracking
# --batch_size 2 --no_aux_loss --eval --resume weights/detr-r50-e632da11.pth
# --coco_path /media/user/data/eslam/kitti_tracking_dataset/custom --dataset_file "kitti_tracking"
# 4) Kitti-Tracking Moving only
# --batch_size 2 --no_aux_loss --eval --resume weights/detr-r50-e632da11.pth
# --coco_path /media/user/data/eslam/kitti_tracking_dataset/moving --dataset_file "kitti_tracking"
# 5) Kitti-old Moving only
# --coco_path /media/user/x_2/proj3054_moving_instance_segmentation/data/kitti --dataset_file "kitti_old"
# --batch_size 2 --no_aux_loss --eval --resume log_baseline_kitti_old/checkpoint_best.pth

# For Training, put configuration as follows:
# 1) coco
# --coco_path /media/user/x_2/coco2017 --output_dir ./log/ --batch_size 6
# 2) cityscapes
# --coco_path /media/user/data/eslam/CityScapes_Dataset/leftImg8bit_trainvaltest --dataset_file "cityscapes"
# --resume weights/detr-r50-e632da11.pth --output_dir ./log/
# 3) Kitti-Tracking static and moving
# --coco_path /media/user/data/eslam/kitti_tracking_dataset/custom --dataset_file "kitti_tracking"
# --resume weights/detr-r50-e632da11.pth --output_dir ./log/
# 4) Kitti-Tracking Moving only
# --coco_path /media/user/data/eslam/kitti_tracking_dataset/moving --dataset_file "kitti_tracking"
# --resume weights/detr-r50-e632da11.pth --output_dir ./log/
# 5) Kitti-old Moving only
# --coco_path /media/user/x_2/proj3054_moving_instance_segmentation/data/kitti --dataset_file "kitti_old"
# --resume weights/detr-r50-e632da11.pth --output_dir ./log_kitti_old/
# 6) instance-segmentation
# --masks --epochs 25 --lr_drop 15 --coco_path /path/to/coco --dataset_file "coco"
# --frozen_weights /output/path/box_model/checkpoint.pth --output_dir /output/path/segm_model
# --masks --epochs 25 --lr_drop 15 --coco_path /media/user/x_2/proj3054_moving_instance_segmentation/data/kitti --dataset_file "kitti_old" --output_dir ./log_del/ --batch_size 4 --frozen_weights experiments/kitti_old/log_kitti_old_baseline_550/checkpoint_best.pth

#--coco_path /media/user/x_2/proj3054_moving_instance_segmentation/data/kitti --dataset_file "kitti_old"  --output_dir ./log_kitti_old_arch#4_concat_480*145/ --dim_feedforward 2048 --batch_size 28 --resume log_coco_arch#4/checkpoint_best.pth


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from config import Config
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096*3, rlimit[1]))


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=True)
        print("Loading pre-trained Done  !!!")
        print("Full model was loaded  !!!")

    output_dir = Path(args.output_dir)
    if args.resume:
        print("Loading pre-trained weights .......")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
            print("Loading pre-trained Done  !!!")
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if args.eval:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            print("Loading pre-trained Done  !!!")
            print("Full model was loaded  !!!")
        else:
            if args.dataset_file == "coco":
                if Config.exp_type == "depth_pos_enc_arch2":
                    for key in list(checkpoint["model"].keys()):
                        if "input_proj" in key:
                            del checkpoint["model"][key]
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                elif Config.exp_type == "shared_rgb_of_N" and not Config.load_full_model and \
                        (Config.variant == "N_B_proj_1_T" or Config.variant == "N_B_N_proj_1_T"
                         or Config.variant == "N_B_N_T_proj_1_T" or (Config.variant == "N_B_N_T" and
                                                                     (Config.sub_variant == "FC" or
                                                                      Config.sub_variant == "NQ_C" or
                                                                      Config.sub_variant == "NQ_C_Decoder"))):
                    for key in list(checkpoint["model"].keys()):
                        if "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "input_proj" in key and Config.variant != "N_B_N_T_proj_1_T":
                            del checkpoint["model"][key]
                    if Config.variant == "N_B_N_T_proj_1_T":
                        for key in list(checkpoint["model"].keys()):
                            if "query_embed" in key:
                                new_key = key.replace("query_embed", "query_embed2")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                            elif "transformer" in key:
                                new_key = key.replace("transformer", "transformer2")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        print("Loading Pre-trained weights is Done for N_B_N_T_proj_1_T")
                    if Config.convert_coco_to1class:
                        del checkpoint["model"]["query_embed.weight"]
                        del checkpoint["model"]["class_embed.weight"]
                        del checkpoint["model"]["class_embed.bias"]
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                elif Config.exp_type == "depth_pos_enc_arch4":
                    for key in list(checkpoint["model"].keys()):
                        if "class_embed" in key:
                            del checkpoint["model"][key]
                        elif "bbox_embed" in key:
                            del checkpoint["model"][key]
                        if "input_proj" in key:
                            new_key = key.replace("input_proj", "input_proj2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "transformer" in key:
                            new_key = key.replace("transformer", "transformer2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "backbone" in key:
                            new_key = key.replace("backbone", "backbone2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model_without_ddp.load_state_dict(checkpoint['model'])
                print("Loading pre-trained Done  !!!")
            if args.dataset_file == "cityscapes" or args.dataset_file == "kitti_tracking"\
                    or args.dataset_file == "kitti_old":
                if not Config.load_full_model:
                    del checkpoint["model"]["class_embed.weight"]
                    del checkpoint["model"]["class_embed.bias"]
                if Config.exp_type == "depth_pos_enc_arch4" and Config.concate:
                    for key in list(checkpoint["model"].keys()):
                        if "class_embed" in key:
                            del checkpoint["model"][key]
                        if "input_proj" in key:
                            new_key = key.replace("input_proj.", "input_proj2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "transformer" in key:
                            new_key = key.replace("transformer.", "transformer2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                elif Config.exp_type == "depth_pos_enc_arch4":
                    for key in list(checkpoint["model"].keys()):
                        if "class_embed" in key:
                            del checkpoint["model"][key]
                        if "input_proj" in key:
                            new_key = key.replace("input_proj.", "input_proj2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "transformer" in key:
                            new_key = key.replace("transformer.", "transformer2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                            new_key = key.replace("transformer.", "transformer3.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                elif Config.exp_type == "shared_rgb_of_N" and not Config.load_full_model \
                        and (Config.variant == "N_B_proj_1_T" or Config.variant == "N_B_N_proj_1_T"):
                    for key in list(checkpoint["model"].keys()):
                        if "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                elif Config.exp_type == "shared_rgb_of_N" and Config.variant == "N_B_N_T_proj_1_T"\
                        and not Config.load_full_model:
                    for name, param in model.named_parameters():
                        print("name = ", name)
                    for key in list(checkpoint["model"].keys()):
                        if "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "transformer" in key:
                            if Config.sharing:
                                new_key = key.replace("transformer", "transformer2")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                            else:
                                for i in range(Config.num_of_repeated_blocks):
                                    new_key = key.replace("transformer.", "transformer." + str(i)+".")
                                    checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        elif "query_embed" in key:
                            for i in range(Config.num_of_repeated_blocks):
                                new_key = key.replace("query_embed.", "query_embed." + str(i)+".")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                    print("Loading Pre-trained weights is Done for N_B_N_T_proj_1_T")
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                elif Config.exp_type == "shared_rgb_of_N" and Config.concate and not Config.load_full_model:
                    for key in list(checkpoint["model"].keys()):
                        if Config.variant == "N_B_N_T" and Config.sub_variant == "NQ_C" and Config.aux_q and Config.sharing:
                            if "class_embed" in key:
                                new_key = key.replace("class_embed", "class_embed_aux_Q")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                            elif "bbox_embed" in key:
                                new_key = key.replace("bbox_embed", "bbox_embed_aux_Q")
                                checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        if "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})

                elif Config.exp_type == "shared_rgb_of_N" and not Config.load_full_model:
                    for key in list(checkpoint["model"].keys()):
                        if "bbox_embed" in key:
                            del checkpoint["model"][key]
                        elif "input_proj" in key:
                            del checkpoint["model"][key]
                        elif "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})

                if Config.exp_type == "depth_pos_enc" or Config.exp_type == "shared_rgb_of"\
                        or Config.exp_type == "cat_4frames_res34":
                    for key in list(checkpoint["model"].keys()):
                        if "input_proj" in key:
                            del checkpoint["model"][key]
                if Config.exp_type == "shared_rgb_of":
                    for key in list(checkpoint["model"].keys()):
                        if "backbone" in key:
                            new_key = key.replace("backbone.", "backbone2.")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                if args.num_queries != 100 and (not Config.load_full_model):
                    del checkpoint["model"]["query_embed.weight"]
                if args.dim_feedforward != 2048 and (not Config.load_full_model):
                    for key in list(checkpoint["model"].keys()):
                        if "transformer" in key or "input_proj" in key or "bbox_embed" in key or "query_embed" in key:
                            del checkpoint["model"][key]
                if Config.exp_type == "depth_pos_enc_arch2" and (not Config.load_full_model):
                    for key in list(checkpoint["model"].keys()):
                        if "transformer.encoder" in key and (not Config.sharing):
                            new_key = key.replace("encoder", "encoder2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        if "transformer.encoder" in key and (Config.variant == "NB_NTE_1T"):
                            new_key = key.replace("encoder", "encoder3")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        if "backbone" in key and (not Config.sharing):
                            new_key = key.replace("backbone", "backbone2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        if "input_proj_transformer" in key and (not Config.sharing):
                            new_key = key.replace("input_proj_transformer", "input_proj_transformer2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})
                        if "input_proj" in key and (not Config.sharing):
                            new_key = key.replace("input_proj", "input_proj2")
                            checkpoint['model'].update({new_key: checkpoint["model"][key]})

                if Config.load_full_model:
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
                else:
                    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                print("Loading pre-trained Done  !!!")
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint\
                and Config.exp_type != "cat_4frames_res34" and Config.exp_type != "depth_pos_enc_arch2"\
                and Config.exp_type != "depth_pos_enc_arch4" and Config.exp_type != "shared_rgb_of_N":
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    best_map = 0
    best_seg_pix_acc = 0
    best_seg_mIoU = 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        """
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        """
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_latest.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # saving best weights
            if Config.seg_task_status:
                if test_stats['seg_mIoU'] > best_seg_mIoU:
                    print("Saving new best seg_mIoU weights.....")
                    best_seg_mIoU = test_stats['seg_mIoU']
                    checkpoint_paths.append(output_dir / 'checkpoint_best_seg_mIoU.pth')
                elif test_stats['seg_pix_ACC'] > best_seg_pix_acc:
                    print("Saving new best seg_pix_ACC weights.....")
                    best_seg_pix_acc = test_stats['seg_pix_ACC']
                    checkpoint_paths.append(output_dir / 'checkpoint_best_seg_pix_ACC.pth')
            if Config.det_task_status:
                if test_stats['coco_eval_bbox'][0] > best_map:
                    print("Saving new best weights.....")
                    best_map = test_stats['coco_eval_bbox'][0]
                    checkpoint_paths.append(output_dir / 'checkpoint_best_det_mAP.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None and not Config.seg_only:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
