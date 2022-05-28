"""
CityScapes dataset which returns image_id for evaluation.

"""
from pathlib import Path
from datasets.coco import CocoDetection, make_coco_transforms


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided CityScapes path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train_images", root / "annotations" / f'{mode}_car_train.json'),
        "val": (root / "val_images", root / "annotations" / f'{mode}_car_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
