"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from datasets.coco import ConvertCocoPolysToMask
import datasets.transforms as T
from pathlib import Path
import torch
import torch.utils.data
import torchvision
from PIL import Image
import os
import os.path
from datasets.augmentation import *
from imagecorruptions import corrupt
from PIL import Image
from matplotlib import cm
from config import Config


def video_transforms(image_set):
    # TODO: should fix data augmentation instead of removing them.
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
            #T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=fixed_image_size),
            T.ToGray(num_output_channels=3),
            normalize,
        ])
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([fixed_image_size], max_size=fixed_image_size),
            T.ToGray(num_output_channels=3),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def load_item(self, idx, aug=False, img_aug_type=None, aug_severity=None):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if aug:
            img = corrupt(np.array(img), corruption_name=img_aug_type, severity=aug_severity)
            img = Image.fromarray(np.uint8(img)).convert('RGB')
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def get_prev_image(self, t_path, img_t, target, back_steps):
        """
        t_path: current frame path.
        img_t: relative frame.
        target: current frame target.
        back_steps: number of backward steps relative to the current frame

        Get prev. image from the given current image path.
        Generic function to get any previous frame given current frame.
        """
        # T-i
        t_1_path = t_path[:-14] + str(int(t_path.split('_')[-1].split('.')[0]) - back_steps).zfill(10) + ".png"
        if os.path.isfile(os.path.join(self.root, t_1_path)):
            img_t_1 = Image.open(os.path.join(self.root, t_1_path)).convert('RGB')
            if self._transforms is not None:
                img_t_1, _ = self._transforms(img_t_1, target)
        else:
            img_t_1 = img_t
        return img_t_1

    def __getitem__(self, idx):
        # Load current image
        img_t, target = self.load_item(idx)

        # Load prev. images
        image_id = self.ids[idx]
        t_path = self.coco.loadImgs(image_id)[0]['file_name']
        # T-1
        img_t_1 = self.get_prev_image(t_path, img_t, target, 1)
        # T-2
        img_t_2 = self.get_prev_image(t_path, img_t_1, target, 2)

        # Debugging
        """
        import cv2
        cv2.imwrite("./1.png", img_t[0].numpy() * 255)
        cv2.imwrite("./2.png", img_t_1[0].numpy() * 255)
        cv2.imwrite("./3.png", img_t_2[0].numpy() * 255)
        """
        # Combine the frames with each others
        img = torch.cat((img_t[0].unsqueeze(dim=0), img_t_1[0].unsqueeze(dim=0), img_t_2[0].unsqueeze(dim=0)), dim=0)
        return img, target


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Tracking path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/images", root / "train" / f'{mode}_car_train.json'),
        "val": (root / "val/images", root / "val" / f'{mode}_car_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=video_transforms(image_set), return_masks=args.masks)
    return dataset
