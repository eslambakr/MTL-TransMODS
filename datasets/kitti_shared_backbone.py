"""
Kitti-Tracking dataset which returns image_id for evaluation.
"""
from pathlib import Path
from datasets.kitti_baseline import video_transforms
from datasets.kitti_stacking import CocoDetection


class DataLoaderSharedBackbone(CocoDetection):
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
        # T-3
        img_t_3 = self.get_prev_image(t_path, img_t_2, target, 3)

        # Debugging
        """
        import cv2
        cv2.imwrite("./1.png", img_t.permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("./2.png", img_t_1.permute(1, 2, 0).numpy() * 255)
        image = cv2.imread("./1.png")
        x_c = target['boxes'][0].numpy()[0] * target['size'][1]
        y_c = target['boxes'][0].numpy()[1] * target['size'][0]
        w = target['boxes'][0].numpy()[2] * target['size'][1]
        h = target['boxes'][0].numpy()[3] * target['size'][0]

        start_point = (x_c-(w/2), y_c-(h/2))
        end_point = (x_c+(w/2), y_c+(h/2))
        color = (0, 0, 0)
        thickness = -1
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imwrite("./1.png", image)
        """

        return img_t, img_t_1, img_t_2, img_t_3, target


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided Kitti-Tracking path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/images", root / "train" / f'{mode}_car_train.json'),
        "val": (root / "val/images", root / "val" / f'{mode}_car_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DataLoaderSharedBackbone(img_folder, ann_file, transforms=video_transforms(image_set),
                                       return_masks=args.masks)
    return dataset
