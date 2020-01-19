# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

import numpy as np 

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.return_feats = cfg.MODEL.ROI_BOX_HEAD.RETURN_FC_FEATS

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, features, targets)
        # use gt as proposals instead of rpn
        proposals = []
        for image_index in range(len(images.image_sizes)):
            image_size = images.image_sizes[image_index]
            image_width = image_size[1]
            image_height = image_size[0]
            image_bboxes = images.image_bboxes[image_index]
            # multiply height & width
            image_bboxes = np.asarray(image_bboxes, dtype='float32')
            # xxyy to xyxy
            image_bboxes = image_bboxes[:,[0,2,1,3]]
            b_row = image_bboxes.shape[0]
            b_col = image_bboxes.shape[1]
            pad_col = b_col
            pad_row = b_row if b_row<100 else 100
            bbox_temp = np.zeros((100,4))
            bbox_temp[:pad_row,:pad_col]= image_bboxes[:pad_row,:pad_col]    
            bbox_temp = torch.from_numpy(bbox_temp)    
            bbox_temp = bbox_temp.cuda()
            #print('bbox', bbox_temp)
            proposal = BoxList(bbox_temp, (image_width,image_height), mode="xyxy")
            proposals.append(proposal)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        if self.return_feats and not self.training:
            return (x, result)

        return result
