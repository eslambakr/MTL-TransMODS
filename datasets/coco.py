# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from config import Config

coco_mapping = {
    1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18,
    20:19, 21:20, 22:21, 23:22, 24:23, 25:24, 27:25, 28:26, 31:27, 32:28, 33:29, 34:30, 35:31, 36:32, 37:33, 38:34
    , 39:35, 40:36, 41:37, 42:38, 43:39, 44:40, 46:41, 47:42, 48:43, 49:44, 50:45, 51:46, 52:47, 53:48, 54:49, 55:50
    , 56:51, 57:52, 58:53, 59:54, 60:55, 61:56, 62:57, 63:58, 64:59, 65:60, 67:61, 70:62, 72:63, 73:64, 74:65, 75:66
    , 76:67, 77:68, 78:69, 79:70, 80:71, 81:72, 82:73, 84:74, 85:75, 86:76, 87:77, 88:78, 89:79, 90:80
}


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if Config.exp_type == "depth_pos_enc_arch2" or Config.exp_type == "depth_pos_enc_arch4":
            return img, img, target
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        device = boxes.device
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        if Config.seg_task_status:
            if Config.convert_coco_to1class:
                # Convert mask to binary mask [No-obj, obj]
                seg_mask = torch.sum(masks, dim=0)
                seg_mask = torch.clamp(seg_mask, 0, 1)
                seg_mask = F.interpolate(seg_mask.unsqueeze(0).unsqueeze(0).float(), size=Config.input_size)
                seg_mask = seg_mask[0][0].type(torch.LongTensor)
                # TODO: Eslam: I will recheck this part as it causes assertion error without using clamp as the target sometimes goes to inf. !!!
                target["seg_masks"] = torch.clamp(seg_mask, 0, 1)
            else:
                # Convert mask according to coco_mapping
                if masks.shape[0] == 0:
                    target["seg_masks"] = torch.zeros(Config.input_size, Config.input_size).long().to(device)
                else:
                    seg_mask = torch.zeros_like(masks)
                    for i, cls in enumerate(classes):
                        """
                        classes[i] = coco_mapping[cls.item()]
                        seg_mask[i] = masks[i] * classes[i]
                        """
                        mapped_class = coco_mapping[cls.item()]
                        seg_mask[i] = masks[i] * mapped_class

                    seg_mask = torch.max(seg_mask, dim=0, keepdim=False).values
                    seg_mask = F.interpolate(seg_mask.unsqueeze(0).unsqueeze(0).float(), size=Config.input_size)
                    seg_mask = seg_mask[0][0].type(torch.LongTensor)
                    # target["seg_masks"] = torch.clamp(seg_mask, 0, 80)
                    target["seg_masks"] = seg_mask

        target["boxes"] = boxes

        if Config.convert_coco_to1class:
            target["labels"] = torch.clamp(classes, 0, 1)
        else:
            #target["labels"] = classes
            mapped_class = torch.zeros_like(classes)
            for i, cls in enumerate(classes):
                mapped_class[i] = coco_mapping[cls.item()]
            target["labels"] = mapped_class

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# Eslam
#"""
def make_coco_transforms(image_set):
    fixed_image_size = Config.input_size
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [fixed_image_size]
    if image_set == 'train':
        if Config.exp_type == "depth_pos_enc_arch2" or Config.exp_type == "depth_pos_enc_arch4":
            return T.Compose([
                # T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=fixed_image_size),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=fixed_image_size),
                #T.RandomSelect(
                #    T.RandomResize(scales, max_size=fixed_image_size),
                #    T.Compose([
                #        T.RandomResize([400, 500, 600]),
                #        T.RandomSizeCrop(384, 600),
                #        T.RandomResize(scales, max_size=fixed_image_size),
                #    ])
                #),
                normalize,
            ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([fixed_image_size], max_size=fixed_image_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


"""
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
"""


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    return_masks = args.masks or Config.seg_task_status
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=return_masks)
    return dataset


if __name__ == "__main__":
    import cv2
    from tqdm import tqdm

    saving_dir = "/media/user/data/eslam/kitti_rstmodseg/"
    image_set = "train"
    root = Path("/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti")
    txt_file = open(saving_dir + "RST_data.txt", "a")
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": (root / "train_rgb", root / "annotations" / "Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations" / "Kitti_MOIS_Test_Annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=None, return_masks=True)
    for i in tqdm(range(len(dataset.ids))):
        image, target = dataset.__getitem__(i)
        merged_mask = torch.sum(target["masks"], dim=0)
        merged_mask = torch.clamp(merged_mask, 0, 1)
        if not (torch.equal(merged_mask.unique().max(), torch.ones_like(merged_mask.unique().max()))):
            if not (torch.equal(merged_mask.unique().max(), torch.zeros_like(merged_mask.unique().max()))):
                print(merged_mask.unique().max())
                print("Error   !!!!")
        image_name = dataset.coco.imgs[i+1]['file_name']
        cv2.imwrite(saving_dir + "masks/" + image_name, merged_mask.cpu().numpy())
        txt_file.write("/data/rgb/" + image_name + "   /data/of/" + image_name + "   /data/masks/" + image_name + "\n")

    print("Done")
