import torch
import torch.nn as nn
import torch.nn.functional as F


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.weight_dict = {'semantic_seg_loss': 100}

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        # Prepare predictions & targets format to match with semantic segmentation Loss:
        preds = tuple(preds['pred_masks'].unsqueeze(0))
        target_list = []
        for tgt in target:
            target_list.append(tgt['seg_masks'].unsqueeze(0))
        if len(target_list) == 1:
            target = target_list[0]
        else:
            target = torch.cat(target_list, dim=0)

        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(semantic_seg_aux_loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(semantic_seg_loss=self._multiple_forward(*inputs))
        else:
            return dict(semantic_seg_loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))
