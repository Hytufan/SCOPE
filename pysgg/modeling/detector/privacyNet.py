# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import datetime
import time

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.pa_roi_heads import build_pa_roi_heads

from pysgg.structures.image_list import to_image_list
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier

class GraphReadout(nn.Module):
    def __init__(self):
        super(GraphReadout, self).__init__()

    def forward(self, x, mask=None):
        graph_representation = x.mean(dim=-2)
        return graph_representation

class PrivacyNet(nn.Module):
    def __init__(self, cfg):
        super(PrivacyNet, self).__init__()
        self.cfg = cfg.clone()
        
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_pa_roi_heads(cfg, self.backbone.out_channels)

        #self.classifier=build_classifier(cfg.PROTECTED.GGNN_D_STATE // 2, self.cfg.EVENT.NUM_CLASSES)
        #self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, images, targets=None, protect_targets=None, logger=None):
        
        images = to_image_list(images)
        #time_1 = time.perf_counter()
        features = self.backbone(images.tensors)
        #time_2 = time.perf_counter()
        self.rpn.eval()
        with torch.no_grad():
            proposals, proposal_losses = self.rpn(images, features, targets)
        #time_3 = time.perf_counter()
        #print("backbone,rpn:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, protect_targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        #time_4 = time.perf_counter()
        
        #print((time_4-time_1)*1000, (time_2-time_1)* 1000, (time_3-time_2)* 1000, (time_4-time_3)* 1000)

        #print("roi:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON and not self.cfg.MODEL.EVENT_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        '''import csv

        # 假设这些时间变量已经定义
        # time_1, time_2, time_3, time_4, time_5

        # 计算时间差
        time_diffs = [
            (time_4 - time_1) * 1000,
            (time_2 - time_1) * 1000,
            (time_3 - time_2) * 1000,
            (time_4 - time_3) * 1000
        ]

        # 指定输出文件路径
        output_file = '/workspace/huangyunyi/res/times2.csv'

        # 将时间差写入CSV文件
        with open(output_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(time_diffs)'''
            
        return result
            
