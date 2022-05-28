#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm

ROOT_DIR = '/media/user/data/eslam/kitti_tracking_dataset/moving/val'
#ROOT_DIR = '/media/eslam/D0FCBC10FCBBEF3A/cityscapes_images/yolact_format/val'
IMAGE_DIR = os.path.join(ROOT_DIR, "images/")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "instances/")

ROOT_DIR = '/media/user/data/eslam/MightyAI_images_all/'
IMAGE_DIR = os.path.join(ROOT_DIR, "val_images/")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "instance_seg_ann")


INFO = {
    "description": "kitti_tracking_dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "eslam_bakr",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'vehicles',
        'supercategory': 'vehicles',
    },
    {
        'id': 2,
        'name': 'person',
        'supercategory': 'person',
    },
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    #print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/vp_fisheye_val_ann.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

#####################################################################################################################
# ########################                           Testing                             ############################
#####################################################################################################################


def test_coco_json_file():
    image_directory = IMAGE_DIR
    annotation_file = ROOT_DIR + '/vp_fisheye_val_ann.json'
    example_coco = COCO(annotation_file)
    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
    category_names = set([category['supercategory'] for category in categories])
    print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))
    category_ids = example_coco.getCatIds(catNms=['square'])
    image_ids = example_coco.getImgIds(catIds=category_ids)
    image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
    print(image_data)
    # load and display instance annotations
    image = io.imread(image_directory + image_data['file_name'])
    plt.imshow(image);
    plt.axis('off')
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    annotations = example_coco.loadAnns(annotation_ids)
    example_coco.showAnns(annotations)


if __name__ == "__main__":
    main()
    test_coco_json_file()
