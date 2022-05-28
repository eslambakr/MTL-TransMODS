# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .cityscapes import build as build_cityscapes
from .kitti_stacking import build as build_kitti_stacking
from .kitti_baseline import build as build_kitti_baseline
from .kitti_shared_backbone import build as build_kitti_shared_backbone
from .kitti_old_rgb_of import build as build_kitti_old_rgb_of
from .kitti_old_baseline import build as build_kitti_old_baseline
from .kitti_old_shared_backbone import build as build_kitti_old_shared_backbone
from .kitti_old_spatial_stacking import build as build_kitti_old_spatial_stacking
from .kitti_old_rgb_of_N import build as build_kitti_old_rgb_of_n
from config import Config


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'cityscapes':
        return build_cityscapes(image_set, args)
    if args.dataset_file == 'kitti_tracking':
        if Config.exp_type == "baseline":
            return build_kitti_baseline(image_set, args)
        elif Config.exp_type == "stacking":
            return build_kitti_stacking(image_set, args)
        elif Config.exp_type == "shared_backbone":
            return build_kitti_shared_backbone(image_set, args)
        else:
            raise NotImplementedError
    if args.dataset_file == 'kitti_old':
        if Config.exp_type == "baseline":
            return build_kitti_old_baseline(image_set, args)
        elif Config.exp_type == "shared_backbone" or Config.exp_type == "depth_pos_enc"\
                or Config.exp_type == "cat_4frames_res34" or Config.exp_type == "depth_pos_enc_arch2"\
                or Config.exp_type == "depth_pos_enc_arch4":
            return build_kitti_old_shared_backbone(image_set, args)
        elif Config.exp_type == "spatial_stacking":
            return build_kitti_old_spatial_stacking(image_set, args)
        elif Config.exp_type == "stacking":
            raise NotImplementedError
        elif Config.exp_type == "shared_rgb_of":
            return build_kitti_old_rgb_of(image_set, args)
        elif Config.exp_type == "shared_rgb_of_N":
            return build_kitti_old_rgb_of_n(image_set, args)
        else:
            raise NotImplementedError
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
