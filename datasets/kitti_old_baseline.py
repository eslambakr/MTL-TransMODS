"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from datasets.coco import CocoDetection
import datasets.transforms as T
from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import os
import os.path
from config import Config


def video_transforms(image_set):
    """
    This transforms shouldn't contain any randomization to make sure it was applied correctly on the temporal images
    with the same effects.
    """
    fixed_image_size = Config.input_size
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [fixed_image_size]
    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=fixed_image_size),
            normalize,
        ])
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([fixed_image_size], max_size=fixed_image_size),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Old dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train_rgb", root / "annotations/Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations/Kitti_MOIS_Test_Annotations.json"),
    }
    img_folder, ann_file = PATHS[image_set]
    return_masks = args.masks or Config.seg_task_status
    dataset = CocoDetection(img_folder, ann_file, transforms=video_transforms(image_set), return_masks=return_masks)
    return dataset
