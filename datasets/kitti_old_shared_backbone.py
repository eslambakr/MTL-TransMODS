"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from datasets.kitti_baseline import video_transforms
from datasets.kitti_stacking import CocoDetection
from PIL import Image
import os
import os.path
from config import Config
from datasets.augmentation import *
from numpy import random
from imagecorruptions import corrupt
from PIL import Image
from matplotlib import cm


class DataLoaderSharedBackbone(CocoDetection):
    def get_prev_image(self, t_path, img_t, target, back_steps, aug=False, img_aug_type=None, aug_severity=None,
                       mapping=None):
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
        t_1_path = str(int(t_path.split('.png')[0]) - back_steps).zfill(6) + ".png"
        if os.path.isfile(os.path.join(self.root, t_1_path)):
            img_t_1 = Image.open(os.path.join(self.root, t_1_path)).convert('RGB')
            if aug:
                img_t_1 = corrupt(np.array(img_t_1), corruption_name=img_aug_type, severity=aug_severity)
                img_t_1 = Image.fromarray(np.uint8(img_t_1)).convert('RGB')
            if self._transforms is not None:
                img_t_1, target = self._transforms(img_t_1, target)

            if Config.aux_q:
                _, target = self.load_item(idx=mapping[t_1_path])
        else:
            valid = False
            img_t_1 = img_t
        if Config.aux_q:
            return img_t_1, target, valid
        return img_t_1, valid

    def __getitem__(self, idx):
        # Augmentation
        img_aug_types = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "motion_blur", "snow",
                         "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
                         "jpeg_compression", "speckle_noise", "spatter"]
        aug_severities = [1, 2, 3, 4]
        aug = False
        img_aug_type = None
        aug_severity = None
        if random_decision(probability=0.75) and Config.augment:
            aug = True
            img_aug_type = random.choice(img_aug_types)
            aug_severity = random.choice(aug_severities)
        # Load current image
        img_t, target = self.load_item(idx, aug, img_aug_type, aug_severity)

        # Load prev. images
        image_id = self.ids[idx]
        t_path = self.coco.loadImgs(image_id)[0]['file_name']
        # T-1
        if Config.aux_q:
            img_t_1, target, valid = self.get_prev_image(t_path, img_t, target, 1, aug, img_aug_type, aug_severity)
        else:
            img_t_1, valid = self.get_prev_image(t_path, img_t, target, 1, aug, img_aug_type, aug_severity)

        if Config.concate or Config.exp_type == "depth_pos_enc" or Config.exp_type == "depth_pos_enc_arch2"\
                or Config.exp_type == "depth_pos_enc_arch4":
            if Config.augment:
                img_t, img_t_1, target = augment(img_t, img_t_1, target)

            # Debugging
            """
            import cv2
            import numpy as np
            import imageio
            imageio.imwrite("/media/user/x_2/del/"+t_path.split('.jpg')[0]+"_1"+".png", img_t.permute(1, 2, 0).numpy() * 255)
            imageio.imwrite("/media/user/x_2/del/"+t_path.split('.jpg')[0]+"_2"+".png", img_t_1.permute(1, 2, 0).numpy() * 255)
            image = cv2.imread("/media/user/x_2/del/"+t_path.split('.jpg')[0]+"_1"+".png")

            if target['boxes'].size(0) != 0:
                x_c = target['boxes'][0].numpy()[0] * target['size'][1]
                y_c = target['boxes'][0].numpy()[1] * target['size'][0]
                w = target['boxes'][0].numpy()[2] * target['size'][1]
                h = target['boxes'][0].numpy()[3] * target['size'][0]

                start_point = (int(x_c-(w/2)), int(y_c-(h/2)))
                end_point = (int(x_c+(w/2)), int(y_c+(h/2)))
                color = (255, 0, 0)
                thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
                imageio.imwrite("/media/user/x_2/del/"+t_path.split('.jpg')[0]+"_1"+".png", image)
            """
            return img_t, img_t_1, target

        # T-2
        if valid:
            img_t_2, valid = self.get_prev_image(t_path, img_t_1, target, 2)
        else:
            return img_t, img_t, img_t, img_t, target
        # T-3
        if valid:
            img_t_3, valid = self.get_prev_image(t_path, img_t_2, target, 3)
        else:
            img_t_3 = img_t_2

        # Debugging
        """
        import cv2
        cv2.imwrite("/media/user/x_2/del/"+t_path+"_1"+".png", img_t.permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("/media/user/x_2/del/"+t_path+"_2"+".png", img_t_1.permute(1, 2, 0).numpy() * 255)
        image = cv2.imread("/media/user/x_2/del/"+t_path+"_1"+".png")
        if target['boxes'].size(0) != 0:
            x_c = target['boxes'][0].numpy()[0] * target['size'][1]
            y_c = target['boxes'][0].numpy()[1] * target['size'][0]
            w = target['boxes'][0].numpy()[2] * target['size'][1]
            h = target['boxes'][0].numpy()[3] * target['size'][0]

            start_point = (x_c-(w/2), y_c-(h/2))
            end_point = (x_c+(w/2), y_c+(h/2))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.imwrite("/media/user/x_2/del/"+t_path+"_1"+".png", image)
        """

        return img_t, img_t_1, img_t_2, img_t_3, target


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Tracking path {root} does not exist'
    PATHS = {
        "train": (root / "train_rgb", root / "annotations/Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations/Kitti_MOIS_Test_Annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataLoaderSharedBackbone(img_folder, ann_file, transforms=video_transforms(image_set),
                                       return_masks=args.masks)
    return dataset
