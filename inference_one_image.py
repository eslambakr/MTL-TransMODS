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
from tqdm import tqdm


# COCO classes
CLASSES = ['Vehicle']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, prob, boxes, img_name):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        """
        cl = p.argmax()
        cl -= 1
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        """
    plt.axis('off')
    plt.savefig(img_name + ".png", dpi=100, pad_inches=0.1, bbox_inches='tight')


def normalize_tensor(in_tensor):
    min_ele = torch.min(in_tensor)
    in_tensor -= min_ele
    in_tensor /= torch.max(in_tensor)
    return in_tensor


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=torch.device(args.device))
    return b


def save_heatmap(heatmap, saving_dir, size, normalized=True):
    if normalized:
        scale = 255
    else:
        scale = 1
    heatmap = cv2.applyColorMap(np.uint8(heatmap*scale), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, size)
    cv2.imwrite(saving_dir, heatmap)


if __name__ == '__main__':
    # Reading Text and Create Mapping dictionary (Id --> Image):
    # ---------------------------------------------------------
    id_img_dict = {}
    with open('datasets/kitti_id_img_mapping.txt') as f:
        lines = f.readlines()
    for line in lines:
        id, img_name = line.split(",")
        id_img_dict[id] = img_name

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # Define model
    model, criterion, postprocessors = build_model(args)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Load pre trained weights
    model_without_ddp = model
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
    print("Loading pre-trained Done  !!!")
    print("Full model was loaded  !!!")

    # Data Loader:
    # -------------
    # mean-std normalize the input image (batch-size: 1)
    #image_name = "000890"
    #org_im = Image.open("/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/test_rgb/" + image_name + ".png")
    if Config.MTL:
        img_w = 512
        img_h = 512
    else:
        img_w = 480
        img_h = 145
    #im = org_im.resize((img_w, img_h))
    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for samples, targets in tqdm(data_loader_val):
        for i, sample in enumerate(samples):
            samples[i] = sample.to(device)
        target = targets[0]

        image_name = id_img_dict[str(target['image_id'].item())].strip()
        org_im = Image.open(
            "/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/test_rgb/" + image_name + ".png")
        im = org_im.resize((img_w, img_h))

        for k in target:
            target[k] = target[k].to(device)
        targets = [target]
        # propagate through the model
        if Config.MTL and Config.shared_dec_shared_q:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > Config.conf_th
        #print(torch.unique(keep, return_counts=True))

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], org_im.size)
        scores = probas[keep]
        if len(bboxes_scaled) == 0:
            continue

        # Draw output:
        #plot_results(org_im, scores, bboxes_scaled, image_name)

        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # propagate through the model
        if Config.MTL and Config.shared_dec_shared_q:
            outputs = model(samples, targets)
        else:
            outputs = model(samples)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0]

        # get the feature map shape
        if Config.MTL:
            h, w = conv_features['3'].tensors.shape[-2:]
        else:
            h, w = conv_features['0'].tensors.shape[-2:]

        i=2
        #bboxes_scaled = torch.cat([bboxes_scaled[0:i], bboxes_scaled[i + 1:]])
        #keep_non_zero = torch.cat([keep.nonzero()[0:i], keep.nonzero()[i + 1:]])
        keep_non_zero = keep.nonzero()
        if Config.MTL and Config.shared_dec_concat_q:
            fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=3, figsize=(22, 4))
        else:
            fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 4))
        colors = COLORS * 100
        dummy = False
        org_im_size = org_im.size
        org_im = np.float32(org_im)
        org_im = cv2.cvtColor(org_im, cv2.COLOR_RGB2BGR)

        if Config.save_att_maps_per_img:
            # Foreground Attention:
            # ----------------------
            heatmap = 0
            for i in range(len(bboxes_scaled)):
                idx = keep_non_zero[i]
                heatmap += dec_attn_weights[0, idx]
            heatmap_foreground = normalize_tensor(heatmap).cpu().detach().view(h, w).numpy()
            save_heatmap(heatmap_foreground, "att_masks_out/Foreground_Attention/" + image_name + ".png", org_im_size)

            # Background Attention:
            # ----------------------
            temp = torch.sum(dec_attn_weights[0], dim=0, keepdim=True)
            temp -= heatmap
            heatmap_background = normalize_tensor(temp).cpu().detach().view(h, w).numpy()
            save_heatmap(heatmap_background, "att_masks_out/Background_Attention/" + image_name + ".png", org_im_size)

        # Save Attention masks per object:
        # ---------------------------------
        if Config.save_att_maps_per_object:
            for i in range(len(bboxes_scaled)):
                idx = keep_non_zero[i]
                if len(bboxes_scaled) > 1:
                    ax_i = axs.T[i]
                else:
                    ax_i = axs.T
                (xmin, ymin, xmax, ymax) = bboxes_scaled[i]
                # ------------------------------------------------------------------------------------------------
                # Save attention maps using opencv:
                heatmap = normalize_tensor(dec_attn_weights[0, idx]).cpu().detach().view(h, w).numpy()
                save_heatmap(heatmap, "att_masks_out/Det_Att_" + image_name + "_" + str(i) + ".png", org_im_size)
                if Config.MTL and Config.shared_dec_shared_q:
                    heatmap = normalize_tensor(dec_attn_weights[0, idx]).cpu().detach().view(h, w).numpy()
                    save_heatmap(heatmap, "att_masks_out/Seg_Att_forground_" + image_name + "_" + str(i) + ".png",
                                 org_im_size)
                    temp = torch.sum(dec_attn_weights[0], dim=0, keepdim=True)
                    temp -= dec_attn_weights[0, idx]
                    heatmap = normalize_tensor(temp).cpu().detach().view(h, w).numpy()
                    save_heatmap(heatmap, "att_masks_out/Seg_Att_background_" + image_name + "_" + str(i) + ".png",
                                 org_im_size)
                else:
                    heatmap = dec_attn_weights[0, 100].cpu().detach().view(h, w).numpy()
                    save_heatmap(heatmap, "att_masks_out/Seg_Att_forground_" + image_name + "_" + str(i) + ".png",
                                 org_im_size)
                    heatmap = dec_attn_weights[0, 101].cpu().detach().view(h, w).numpy()
                    save_heatmap(heatmap, "att_masks_out/Seg_Att_background_" + image_name + "_" + str(i) + ".png",
                                 org_im_size)
                    heatmap = dec_attn_weights[0, 0].cpu().detach().view(h, w)
                    for ii in range(99):
                        if ii + 1 == idx:
                            continue
                        heatmap += dec_attn_weights[0, ii + 1].cpu().detach().view(h, w)
                    save_heatmap(heatmap.numpy(),
                                 "att_masks_out/Seg_Att_background_fromDet._" + image_name + "_" + str(i) + ".png",
                                 org_im_size)

                ax = ax_i[0]
                #dec_attn_weights[0, idx] = normalize_tensor(dec_attn_weights[0, idx])
                ax.imshow(dec_attn_weights[0, idx].cpu().detach().view(h, w))
                #ax.imshow(dec_attn_weights[0, 101].cpu().detach().view(h, w))
                ax.axis('off')
                if Config.MTL and Config.shared_dec_concat_q:
                    ax.set_title(f'Det. Query id: {idx.item()}')
                else:
                    ax.set_title(f'query id: {idx.item()}')
                ax = ax_i[1]
                ax.imshow(im)
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color='blue', linewidth=1))
                ax.axis('off')
                if Config.MTL and Config.shared_dec_concat_q:
                    ax.set_title(f'I/P Image')
                    ax = ax_i[2]
                    ax.imshow(dec_attn_weights[0, 100].cpu().detach().view(h, w))
                    """
                    eslam = dec_attn_weights[0, 0].cpu().detach().view(h, w)
                    for ii in range(99):
                        if ii+1 == idx:
                            continue
                        eslam += dec_attn_weights[0, ii+1].cpu().detach().view(h, w)
                    ax.imshow(eslam)
                    """
                    ax.axis('off')
                    ax.set_title(f'Seg. Query id: {100}')
                #ax.set_title(CLASSES[probas[idx].argmax()])
            fig.tight_layout()
            #plt.savefig("eslam.png", dpi=300, pad_inches=0.1, bbox_inches='tight')

        # ------------------------------------------------------------------------------------------------
        if Config.saving_out_RGB:
            # Save Output (Detection & Segmentation outputs):
            # ------------------------------------------------
            mask_eslam = outputs['pred_masks'][0].permute(1, 2, 0).argmax(-1).unsqueeze(-1) * 255
            mask_eslam = torch.cat([mask_eslam, torch.zeros_like(mask_eslam), torch.zeros_like(mask_eslam)], -1)
            mask_eslam = mask_eslam.data.cpu().numpy()
            mask_eslam = cv2.resize(src=np.float32(mask_eslam), dsize=org_im_size)
            alpha = 0.5
            beta = 0.9
            overlayed = cv2.addWeighted(src1=np.float32(mask_eslam), alpha=alpha, src2=np.float32(org_im), beta=beta, gamma=0)
            cv2.imwrite("att_masks_out/seg_out_only/" + image_name + ".png", overlayed)

            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > Config.conf_th
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], org_im_size)
            for i in range(len(bboxes_scaled)):
                (xmin, ymin, xmax, ymax) = bboxes_scaled[i]
                detection_only_img = cv2.rectangle(np.float32(org_im), (xmin, ymin), (xmax, ymax),
                                                   color=(0, 255, 0), thickness=2)
                overlayed = cv2.rectangle(overlayed, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
            cv2.imwrite("att_masks_out/det_out_only/" + image_name + ".png", detection_only_img)
            cv2.imwrite("att_masks_out/det_seg_out/" + image_name + ".png", overlayed)
        # ------------------------------------------------------------------------------------------------
