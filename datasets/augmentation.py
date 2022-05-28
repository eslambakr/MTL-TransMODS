import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt
from imagecorruptions import corrupt


######################################################################################
def random_decision(probability):
    return random.random() < probability


def convert_tensor_rgb(tensor):
    return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


def convert_rgb_tensor(rgb_img):
    return torch.from_numpy(rgb_img.astype(np.float32)).permute(2, 0, 1)


def convert_rgb_to_float(image):
    return image.astype(np.float32)


def convert_to_img_coords(target):
    boxes = target['boxes']
    width = target['size'][1]
    height = target['size'][0]
    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height
    target['boxes'] = boxes
    return target


def normalize_to_img_coords(target):
    boxes = target['boxes']
    width = target['size'][1]
    height = target['size'][0]
    boxes[:, 0] /= width
    boxes[:, 2] /= width
    boxes[:, 1] /= height
    boxes[:, 3] /= height
    target['boxes'] = boxes
    return target


def random_saturation(image, lower=0.5, upper=1.5):
    assert upper >= lower, "contrast upper must be >= lower."
    assert lower >= 0, "contrast lower must be non-negative."
    image[:, :, 1] *= random.uniform(lower, upper)
    return image


def random_hue(image, delta=18.0):
    assert 0.0 <= delta <= 360.0
    image[:, :, 0] += random.uniform(-delta, delta)
    image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    return image


def convert_color(image, conversion_type):
    if conversion_type == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif conversion_type == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def random_mirror(image1, target, image2=None):
    # Horizontal mirroring
    boxes = target['boxes']
    if not boxes.size()[0]:
        if image2 is None:
            return image1, target
        else:
            return image1, image2, target
    width = target['size'][1]
    # Mirror image
    image1 = image1[:, ::-1, :]
    if image2 is not None:
        image2 = image2[:, ::-1, :]
    # Mirror Target
    x1 = boxes[:, 0] - (boxes[:, 2] / 2)
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)
    x1 = width - x1
    x2 = width - x2
    boxes[:, 0] = x1 - (boxes[:, 2] / 2)
    boxes[:, 1] = y1 + (boxes[:, 3] / 2)
    target['boxes'] = boxes
    if image2 is None:
        return image1, target
    else:
        return image1, image2, target


def random_flip(image, target):
    boxes = target['boxes']
    height = target['size'][0]
    image = image[::-1, :]
    boxes = boxes.copy()
    boxes[:, 1::2] = height - boxes[:, 3::-2]
    target['boxes'] = boxes
    return image, target


def augment(image1, image2, target):
    img_aug_types = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                     "motion_blur", "zoom_blur", "snow", "frost", "fog",
                     "brightness", "contrast", "elastic_transform", "pixelate",
                     "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate"]
    aug_severities = [1, 2, 3, 4, 5]
    # Horizontal Mirror
    if random_decision(probability=0):
        target = convert_to_img_coords(target)
        image1 = convert_tensor_rgb(image1)
        image2 = convert_tensor_rgb(image2)
        image1, image2, target = random_mirror(image1, target, image2)
        image1 = convert_rgb_tensor(image1)
        image2 = convert_rgb_tensor(image2)
        target = normalize_to_img_coords(target)

    # Augment images
    if random_decision(probability=0.7) and False:
        img_aug_type = random.choice(img_aug_types)
        aug_severity = random.choice(aug_severities)
        image1 = convert_tensor_rgb(image1).astype(np.uint8)*255
        image2 = convert_tensor_rgb(image2).astype(np.uint8)*255
        image1 = corrupt(image1, corruption_name=img_aug_type, severity=aug_severity)
        image2 = corrupt(image2, corruption_name=img_aug_type, severity=aug_severity)
        image1 = convert_rgb_tensor(image1)/255
        image2 = convert_rgb_tensor(image2)/255

    return image1, image2, target


if __name__ == '__main__':
    print("still in progress")

