"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from PIL import Image
import os
from datasets.kitti_baseline import video_transforms
from datasets.kitti_stacking import CocoDetection
import datasets.transforms as T
from config import Config


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


class DataLoaderRGBOF(CocoDetection):
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

    def __getitem__(self, idx):
        # Load current image
        img_t, target = self.load_item(idx)

        image_id = self.ids[idx]
        t_path = self.coco.loadImgs(image_id)[0]['file_name']
        # Optical Flow
        optical_flow = self.get_of(t_path, target)

        # Debugging
        """
        import cv2
        cv2.imwrite("./1.png", img_t.permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("./2.png", optical_flow.permute(1, 2, 0).numpy() * 255)
        image = cv2.imread("./1.png")
        x_c = target['boxes'][0].numpy()[0] * target['size'][1]
        y_c = target['boxes'][0].numpy()[1] * target['size'][0]
        w = target['boxes'][0].numpy()[2] * target['size'][1]
        h = target['boxes'][0].numpy()[3] * target['size'][0]

        start_point = (x_c-(w/2), y_c-(h/2))
        end_point = (x_c+(w/2), y_c+(h/2))
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imwrite("./1.png", image)
        """

        return img_t, optical_flow, target


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Old dataset path {root} does not exist'
    PATHS = {
        "train": (root / "train_rgb", root / "annotations/Kitti_MOIS_Train_Annotations.json"),
        "val": (root / "test_rgb", root / "annotations/Kitti_MOIS_Test_Annotations.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataLoaderRGBOF(img_folder, ann_file, transforms=video_transforms(image_set), return_masks=args.masks)
    return dataset
