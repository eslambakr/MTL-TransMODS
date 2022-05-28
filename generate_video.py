from models import build_model
from main import get_args_parser
import argparse
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import datasets.transforms as Tdetr
from config import Config
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import torch
from PIL import Image
import glob
from tqdm import tqdm
from inference_one_image import save_heatmap
import os


def generate_empty_att_maps():
    """
    generate empty attention maps for empty images that don't contain any outputs
    """
    for i in range(56):
        att_map = torch.randint(150, 255, (1242, 375))
        image_name = 8267
        save_heatmap(att_map, "att_masks_out/dummy_Attention/" + str(image_name+i).zfill(6) + ".png",
                     (1242, 375), normalized=False)


def list_images_in_dir(image_dir, image_extension, zfill=True):
    image_names = []
    image_list = []
    if zfill:
        for filename in glob.glob(image_dir + '/*.' + image_extension):
            image_names.append(filename)
        image_names = sorted(image_names)
        image_names.sort()
    else:
        images = [img.split('.')[0] for img in os.listdir(image_dir) if img.endswith("."+image_extension)]
        images.sort(key=int)
        for img in images:
            image_names.append(image_dir + img + ".jpg")
    for name in image_names:
        im = cv2.imread(name)
        image_list.append(im)
    return image_list


def concate_2_imgs_V(src1_dir, src2_dir, image_extension):
    src1_images = list_images_in_dir(src1_dir, image_extension)
    src2_images = list_images_in_dir(src2_dir, image_extension)
    for i in range(len(src1_images)):
        im_v = cv2.vconcat([src1_images[i], src2_images[i]])
        cv2.imwrite('att_masks_out/S3/out_and_foreatt/'+str(i)+'.jpg', im_v)


def concate_4_imgs_H(src_dir, image_extension):
    src_images = list_images_in_dir(src_dir, image_extension)
    im_h = cv2.vconcat([src_images[0], src_images[1], src_images[2], src_images[3]])
    cv2.imwrite('att_masks_out/S2/out_and_foreatt/Untitled Folder/' + '.jpg', im_h)


def generate_video_from_imgs(src_dir, image_extension, video_name):
    images = list_images_in_dir(src_dir, image_extension, zfill=False)
    video_name = video_name + '.avi'
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    # concate_2_imgs_V("att_masks_out/S3/out/", "att_masks_out/S3/fore_att/", "png")
    generate_video_from_imgs("att_masks_out/S1/out_and_foreatt/", "jpg", "S1")
