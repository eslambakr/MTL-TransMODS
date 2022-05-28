"""
Kitti dataset which returns N number of RGB + OF frames.
"""
from pathlib import Path
from PIL import Image
import os
from datasets.kitti_baseline import video_transforms
from datasets.kitti_old_shared_backbone import DataLoaderSharedBackbone
import datasets.transforms as T
from config import Config
from numpy import random
from imagecorruptions import corrupt
import numpy as np


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


class DataLoaderRGBOFN(DataLoaderSharedBackbone):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super().__init__(img_folder, ann_file, transforms, return_masks)
        self.mapping_dict = {}
        # Creating mapping from image name to image idx
        for i in range(len(self.coco.imgs)):
            item = self.coco.imgs[i + 1]
            self.mapping_dict[item['file_name']] = item['id'] - 1

    def get_of(self, t_path, target):
        """
        t_path: current frame path.
        target: current frame target.

        Get corresponding Optical Flow image.
        """
        # OF
        if self.root.name == "train_rgb":
            of_name = "train_OF"
        elif self.root.name == "test_rgb":
            of_name = "test_OF"
        of = Image.open(os.path.join(self.root.parent, of_name, t_path)).convert('RGB')
        if self._transforms is not None:
            of, _ = self._transforms(of, target)

        return of

    def get_prev_of(self, t_path, img_t, target, back_steps, aug=False, img_aug_type=None, aug_severity=None):
        """
        t_path: current frame path.
        img_t: relative frame.
        target: current frame target.
        back_steps: number of backward steps relative to the current frame

        Get prev. image from the given current image path.
        Generic function to get any previous frame given current frame.
        """
        # T-i
        valid = True
        if self.root.name == "train_rgb":
            of_name = "train_OF"
        elif self.root.name == "test_rgb":
            of_name = "test_OF"
        t_1_path = str(int(t_path.split('.png')[0]) - back_steps).zfill(6) + ".png"
        if os.path.isfile(os.path.join(self.root.parent, of_name, t_1_path)):
            img_t_1 = Image.open(os.path.join(self.root.parent, of_name, t_1_path)).convert('RGB')
            if aug:
                img_t_1 = corrupt(np.array(img_t_1), corruption_name=img_aug_type, severity=aug_severity)
                img_t_1 = Image.fromarray(np.uint8(img_t_1)).convert('RGB')
            if self._transforms is not None:
                img_t_1, _ = self._transforms(img_t_1, target)
        else:
            valid = False
            img_t_1 = img_t
        return img_t_1, valid

    def __getitem__(self, idx):
        # Load current image
        img_t, target = self.load_item(idx)

        # Load prev. images
        image_id = self.ids[idx]
        t_path = self.coco.loadImgs(image_id)[0]['file_name']
        of = self.get_of(t_path, target)
        valid = True
        if Config.aux_q:
            input_data = [img_t, of, target]
        else:
            input_data = [img_t, of]
        for i in range(1, Config.num_of_repeated_blocks):
            img_t_i = img_t
            of_t_i = of
            # T-i
            if valid:
                if Config.aux_q:
                    img_t_i, target, valid = self.get_prev_image(t_path, img_t_i, target, i, mapping=self.mapping_dict)
                else:
                    img_t_i, valid = self.get_prev_image(t_path, img_t_i, target, i)
                of_t_i, _ = self.get_prev_of(t_path, of_t_i, target, i)
            input_data.append(img_t_i)
            input_data.append(of_t_i)
            if Config.aux_q:
                input_data.append(target)

        if not Config.aux_q:
            input_data.append(target)
        # Debugging
        """
        import cv2
        cv2.imwrite("./1.png", img_t.permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("./2.png", of.permute(1, 2, 0).numpy() * 255)
        image = cv2.imread("./1.png")
        image = cv2.imread("/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/test_rgb/" + t_path)
        if target['boxes'].size(0) != 0:
            for i in range(target['boxes'].size(0)):
                x_c = target['boxes'][i].numpy()[0] * 1242 # target['size'][1]
                y_c = target['boxes'][i].numpy()[1] * 375 # target['size'][0]
                w = target['boxes'][i].numpy()[2] * 1242 # target['size'][1]
                h = target['boxes'][i].numpy()[3] * 375 # target['size'][0]

                start_point = (int(x_c-(w/2)), int(y_c-(h/2)))
                end_point = (int(x_c+(w/2)), int(y_c+(h/2)))
                color = (255, 0, 0)
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.imwrite("./3.png", image)
        """

        return input_data


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Old dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train_rgb", root / "annotations/Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations/Kitti_MOIS_Test_Annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    return_masks = args.masks or Config.seg_task_status
    dataset = DataLoaderRGBOFN(img_folder, ann_file, transforms=video_transforms(image_set), return_masks=return_masks)
    return dataset
