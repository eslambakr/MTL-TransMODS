"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from datasets.kitti_baseline import video_transforms
from datasets.kitti_stacking import CocoDetection
from datasets.kitti_old_shared_backbone import DataLoaderSharedBackbone
from PIL import Image
import os
import os.path
from config import Config
import torch


class DataLoaderSpatialBackbone(DataLoaderSharedBackbone):
    def __getitem__(self, idx):
        # Load current image
        img_t, target = self.load_item(idx)

        # Load prev. images
        image_id = self.ids[idx]
        t_path = self.coco.loadImgs(image_id)[0]['file_name']
        # T-1
        img_t_1, valid = self.get_prev_image(t_path, img_t, target, 1)

        img = torch.cat((img_t, img_t_1), dim=1)

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

        return img, target


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Tracking path {root} does not exist'
    PATHS = {
        "train": (root / "train_rgb", root / "annotations/Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations/Kitti_MOIS_Test_Annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataLoaderSpatialBackbone(img_folder, ann_file, transforms=video_transforms(image_set),
                                        return_masks=args.masks)
    return dataset
