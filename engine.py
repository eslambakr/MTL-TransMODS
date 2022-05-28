# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import time
import torch
import logging

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from config import Config
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models.semantic_seg.seg_metric import SegmentationMetric


def plot_attention_head(in_tokens, translated_tokens, attention, X_label, Ylabel, saving_name):

    ax = plt.gca()
    ax.matshow(attention)
    plt.xlabel(X_label, size=10)
    plt.ylabel(Ylabel, size=10)
    plt.xticks(size=1)
    plt.yticks(size=1)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    ax.set_xticklabels(in_tokens, rotation=90)
    ax.set_yticklabels(translated_tokens)
    plt.savefig(saving_name, dpi=1200, pad_inches=0.1, bbox_inches='tight')


# Eslam: Calculate the total FPS
def average(lst):
    return sum(lst) / len(lst)


CLASSES = ['Moving Vehicle']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox.cpu())
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(img, prob, boxes, unique_counter):
    img = img.decompose()[0]
    #img = img[0].decompose()[0]
    img = img.cpu().numpy()[0].transpose(1,2,0)
    # Un-Normalize image
    img *= [0.229, 0.224, 0.225]
    img += [0.485, 0.456, 0.406]
    img *= 255
    img = img.astype(np.float32)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)

    if boxes.nelement() == 0:
        return
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        cl = p.argmax()
        #text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
    cv2.imwrite(Config.saving_det_out_dir+str(unique_counter)+".png", img)


def Merge(dict1, dict2):
    return(dict2.update(dict1))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    if Config.MTL:
        for cr in criterion:
            cr.train()
    else:
        criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not Config.seg_only:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if (Config.exp_type == "shared_backbone" or Config.exp_type == "shared_rgb_of"
            or Config.exp_type == "depth_pos_enc" or Config.exp_type == "cat_4frames_res34"
            or Config.exp_type == "depth_pos_enc_arch2" or Config.exp_type == "depth_pos_enc_arch4"
            or Config.exp_type == "shared_rgb_of_N")\
                and not Config.pre_training_coco:
            for i, sample in enumerate(samples):
                samples[i] = sample.to(device)
        else:
            samples = samples.to(device)

        if Config.aux_q and not Config.pre_training_coco:
            all_target = []
            for block in range(Config.num_of_repeated_blocks):
                block_targets = targets[block]
                new_target = []
                for i, target in enumerate(block_targets):  # Loop on batch size
                    converted_dict = {}
                    for k, v in target.items():
                        converted_dict[k] = v.to(device)
                    new_target.append(converted_dict)
                all_target.append(new_target)
            targets = all_target
        else:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Debugging
        """
        import cv2
        cv2.imwrite("./1.png", samples[0].decompose()[0][0].permute(1, 2, 0).cpu().numpy() * 255)
        cv2.imwrite("./2.png", samples[1].decompose()[0][0].permute(1, 2, 0).cpu().numpy() * 255)
        cv2.imwrite("./3.png", samples[2].decompose()[0][0].permute(1, 2, 0).cpu().numpy() * 255)
        cv2.imwrite("./4.png", samples[3].decompose()[0][0].permute(1, 2, 0).cpu().numpy() * 255)
        """

        if Config.aux_q:
            losses_aux_q = 0
            loss_aux_q_dict = {}
            outputs, out_aux_intermediate_q = model(samples)
            for i in range(len(out_aux_intermediate_q)):
                if Config.pre_training_coco:
                    loss_dict = criterion(out_aux_intermediate_q[i], targets)
                else:
                    loss_dict = criterion(out_aux_intermediate_q[i], targets[i])
                weight_dict = criterion.weight_dict.copy()  # True copy
                for k in loss_dict.keys():
                    loss_dict[k + "_aux_q_" + str(i)] = loss_dict.pop(k)
                for k in weight_dict.keys():
                    weight_dict[k + "_aux_q_" + str(i)] = weight_dict.pop(k)
                losses_aux_q += sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                Merge(loss_dict, loss_aux_q_dict)  # update loss_aux_q_dict
            if Config.pre_training_coco:
                loss_dict = criterion(outputs, targets)
            else:
                loss_dict = criterion(outputs, targets[0])
            weight_dict = criterion.weight_dict
        else:
            if Config.shared_dec_shared_q:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            if Config.MTL:
                loss_dict = {}
                weight_dict = {}
                for cr in criterion:
                    loss_dict.update(cr(outputs, targets))
                    weight_dict.update(cr.weight_dict)
            else:
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if Config.aux_q:
            Merge(loss_aux_q_dict, loss_dict)  # update loss_dict
            losses += losses_aux_q

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if not Config.seg_only:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    if Config.MTL:
        for cr in criterion:
            cr.eval()
    else:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not Config.seg_only:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    if Config.seg_task_status:
        sematic_seg_metric = SegmentationMetric(nclass=Config.num_classes)
        sematic_seg_metric.reset()
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # Eslam: Calculate the total FPS
    total_inf_time = []
    unique_counter = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        if (Config.exp_type == "shared_backbone" or Config.exp_type == "shared_rgb_of"
            or Config.exp_type == "depth_pos_enc" or Config.exp_type == "cat_4frames_res34"
            or Config.exp_type == "depth_pos_enc_arch2" or Config.exp_type == "depth_pos_enc_arch4"
            or Config.exp_type == "shared_rgb_of_N")\
                and not Config.pre_training_coco:
            for i, sample in enumerate(samples):
                samples[i] = sample.to(device)
        else:
            samples = samples.to(device)

        if Config.aux_q and not Config.pre_training_coco:
            all_target = []
            for block in range(Config.num_of_repeated_blocks):
                block_targets = targets[block]
                new_target = []
                for i, target in enumerate(block_targets):  # Loop on batch size
                    converted_dict = {}
                    for k, v in target.items():
                        converted_dict[k] = v.to(device)
                    new_target.append(converted_dict)
                all_target.append(new_target)
            targets = all_target
        else:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if Config.saving_attention_map:
            # Attention Maps:
            # use lists to store the outputs via up-values
            conv_features, enc_attn_weights, dec_attn_weights = [], [], []
            hooks = [model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
                     model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                         lambda self, input, output: enc_attn_weights.append(output[1])),
                     model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                         lambda self, input, output: dec_attn_weights.append(output[1])), ]
        # Eslam: Calculate the total FPS
        start_time = time.time()

        if Config.aux_q:
            outputs, out_aux_intermediate_q = model(samples)
            if not Config.pre_training_coco:
                targets = targets[0]
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
        else:
            if Config.shared_dec_shared_q:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            if Config.MTL:
                loss_dict = {}
                weight_dict = {}
                for cr in criterion:
                    loss_dict.update(cr(outputs, targets))
                    weight_dict.update(cr.weight_dict)
            else:
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
            """
            if unique_counter == 23:
                output_eslam = cv2.imread("/media/user/x_2/proj3054_moving_instance_segmentation/data/kitti/test_rgb/003272.png")
                mask_eslam = outputs['pred_masks'][0].permute(1, 2, 0).argmax(-1).unsqueeze(-1)*255
                mask_eslam = torch.cat([mask_eslam, torch.zeros_like(mask_eslam), torch.zeros_like(mask_eslam)],-1)
                mask_eslam = mask_eslam.data.cpu().numpy()
                mask_eslam = cv2.resize(src=np.float32(mask_eslam), dsize=(output_eslam.shape[1], output_eslam.shape[0]))
                overlayed = cv2.addWeighted(np.float32(mask_eslam), 0.5, np.float32(output_eslam), 1 - 0.5, 0, np.float32(output_eslam))
                cv2.imwrite("/home/user/masks_check/seg_only.png", overlayed)

                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.9
                bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep],
                                               (output_eslam.shape[1], output_eslam.shape[0]))
                (xmin, ymin, xmax, ymax) = bboxes_scaled[0]
                detection_only_img = cv2.rectangle(np.float32(output_eslam), (xmin, ymin), (xmax, ymax),
                                                   color=(0, 255, 0), thickness=2)
                cv2.imwrite("/home/user/masks_check/det_only.png", detection_only_img)
                overlayed = cv2.rectangle(overlayed, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                cv2.imwrite("/home/user/masks_check/det_seg.png", overlayed)
                cv2.imwrite("/home/user/masks_check/in"+ str(unique_counter)+".png",
                            cv2.cvtColor(samples[0].decompose()[0][0].permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR))
                cv2.imwrite("/home/user/masks_check/pred"+ str(unique_counter)+".png",
                            outputs['pred_masks'][0].permute(1, 2, 0).argmax(-1).data.cpu().numpy()*255)
                cv2.imwrite("/home/user/masks_check/gt"+ str(unique_counter)+".png",
                            targets[0]['seg_masks'].cpu().numpy()*255)
            """

        total_inf_time.append(time.time() - start_time)

        if Config.saving_det_out:
            # keep only predictions with 0.7+ confidence
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.9
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], [Config.input_size, Config.input_size])
            #plot_results(samples, probas[keep], bboxes_scaled, unique_counter)

        if Config.saving_attention_map:
            # Attention Maps:
            for hook in hooks:
                hook.remove()
            # don't need the list anymore
            conv_features = conv_features[0]
            enc_attn_weights = enc_attn_weights[0][0]  # must run the code using batch_size=1.
            enc_attn_weights = enc_attn_weights.cpu().numpy()
            plot_attention_head(np.arange(enc_attn_weights.shape[1]), np.arange(enc_attn_weights.shape[0]),
                                enc_attn_weights, "HW", "HW", 'encoder attention map.png')
            dec_attn_weights = dec_attn_weights[0]
            dec_attn_weights_np = dec_attn_weights[0]
            dec_attn_weights_np = dec_attn_weights_np.cpu().numpy()
            plot_attention_head(np.arange(dec_attn_weights_np.shape[1]), np.arange(dec_attn_weights_np.shape[0]),
                                dec_attn_weights_np, "HW", "N_q", 'decoder attention map.png')
            plot_attention_head(np.arange(dec_attn_weights_np.shape[1]), [25,26,27,28,29],
                                dec_attn_weights_np[25:30, :], "HW", "N_q", 'decoder attention map2.png')
            plot_attention_head(np.arange(dec_attn_weights_np.shape[1]), [40, 41, 42, 43, 44],
                                dec_attn_weights_np[40:45, :], "HW", "N_q", 'decoder attention map3.png')
            plot_attention_head(np.arange(dec_attn_weights_np.shape[1]), [67, 68, 69, 70, 71,72,73,74],
                                dec_attn_weights_np[67:75, :], "HW", "N_q", 'decoder attention map4.png')
            plot_attention_head(np.arange(dec_attn_weights_np.shape[1]), [96, 97, 98, 99],
                                dec_attn_weights_np[96:, :], "HW", "N_q", 'decoder attention map5.png')

            # get the feature map shape
            h, w = conv_features['0'].tensors.shape[-2:]
            if Config.MTL:
                h, w = conv_features['3'].tensors.shape[-2:]

            fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
            colors = COLORS * 100
            obj_counter = 0
            for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
                # dec_attn_weights
                if os.path.isfile(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png"):
                    att_map_img = cv2.imread(
                        Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png", 0).astype(np.float32)/255
                    att_map_img += dec_attn_weights[0, idx].view(h, w).cpu().numpy().astype(np.float32)
                    cv2.imwrite(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png",
                                att_map_img*255)
                else:
                    cv2.imwrite(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png",
                                dec_attn_weights[0, idx].view(h, w).cpu().numpy()*255)

                """
                cv2.imwrite(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + "_" + str(
                    obj_counter) + ".png",
                            dec_attn_weights[0, idx].view(h, w).cpu().numpy() * 255)
                att_map_img = cv2.imread(
                    Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + "_" + str(
                        obj_counter) + ".png", 0)
                att_map_img = ((att_map_img / att_map_img.max())*255).astype(np.uint8)
                att_map_img = cv2.applyColorMap(att_map_img, cv2.COLORMAP_HOT)
                att_map_img = cv2.resize(att_map_img, (Config.input_size, Config.input_size))
                cv2.imwrite(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + "_" + str(obj_counter) + ".png",
                            att_map_img)
                obj_counter += 1
                """
            att_map_img = cv2.imread(
                Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png", 0)
            if att_map_img is not None:
                att_map_img = ((att_map_img / att_map_img.max()) * 255).astype(np.uint8)
                att_map_img = cv2.applyColorMap(att_map_img, cv2.COLORMAP_HOT)
                att_map_img = cv2.resize(att_map_img, (Config.input_size, Config.input_size))
                cv2.imwrite(Config.saving_attention_map_dir + "dec_attn_weights/" + str(unique_counter) + ".png",
                            att_map_img)

            #fig.tight_layout()
        unique_counter += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if Config.seg_task_status:
            target_list = []
            for tgt in targets:
                target_list.append(tgt['seg_masks'].unsqueeze(0))
            if len(target_list) == 1:
                seg_targets = target_list[0]
            else:
                seg_targets = torch.cat(target_list, dim=0)
            sematic_seg_metric.update(outputs['pred_masks'], seg_targets)
            pixAcc, mIoU, category_iou = sematic_seg_metric.get(return_category_iou=True)
            #print("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(unique_counter + 1, pixAcc * 100, mIoU * 100))
            if Config.seg_only:
                continue

        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # Eslam: Calculate the total FPS
    print("Total FPS = ", 1/average(total_inf_time))

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if Config.seg_task_status:
        pixAcc, mIoU = sematic_seg_metric.get()
        print("[EVAL END] Total for this Epoch: pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))
        stats['seg_pix_ACC'] = pixAcc
        stats['seg_mIoU'] = mIoU
        if Config.seg_only:
            return stats, None

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator
