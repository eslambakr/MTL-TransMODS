from pathlib import Path

import torch
import torch.utils.data
import torchvision
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as coco_mask

import datasets.transforms as T
from config import Config


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, data_type, count_start=0):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.images_count = count_start
        self.saving_dir = "/media/user/x_2/kitti_txt_format/"
        self.images_name_file = open(self.saving_dir + data_type + ".txt", "w")

    def __getitem__(self, idx):
        img, targets = super(CocoDetection, self).__getitem__(idx)
        if len(targets) == 0:
            return
        resized_img = cv2.resize(np.array(img), (416, 416))
        cv2.imwrite(self.saving_dir + "images/" + str(self.images_count) + ".png", resized_img)
        file = open(self.saving_dir + "labels/" + str(self.images_count) + ".txt", "w")
        for tg in targets:
            # Convert from upper point to center point
            tg['bbox'][0] += tg['bbox'][2]/2
            tg['bbox'][1] += tg['bbox'][3]/2

            # Normalize labels
            tg['bbox'][0] /= img.width
            tg['bbox'][2] /= img.width
            tg['bbox'][1] /= img.height
            tg['bbox'][3] /= img.height
            file.write("0 " + str(tg['bbox'][0]) + " " + str(tg['bbox'][1]) + " " + str(tg['bbox'][2]) + " " +
                       str(tg['bbox'][3]) + "\n")
        file.close()
        self.images_count += 1
        self.images_name_file.write(self.saving_dir + "images/" + str(self.images_count) + ".png\n")
        return

    def convert_data(self):
        for idx in tqdm(range(len(self.ids))):
            self.__getitem__(idx)


if __name__ == "__main__":
    print("Convert Training files ....")
    train_imgs_dir = "/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/train_rgb/"
    train_ann_file = "/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/annotations/Kitti_MOIS_Train_Annotations.json"
    test_imgs_dir = "/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/test_rgb/"
    test_ann_file = "/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/annotations/Kitti_MOIS_Test_Annotations.json"
    data_loader = CocoDetection(train_imgs_dir, train_ann_file, "kitti_train")
    data_loader.convert_data()

    print("Convert Val files ....")
    val_data_loader = CocoDetection(test_imgs_dir, test_ann_file, "kitti_val", count_start=data_loader.images_count)
    val_data_loader.convert_data()
