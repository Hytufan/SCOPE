# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from .attribute_head.attribute_head import build_roi_attribute_head
from .box_head.box_head import build_roi_box_head
from .relation_head.pa_event_head import build_pa_roi_event_head

import datetime
import time


class CombinedPAROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedPAROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        self.protected_area=None
        self.scene_recognizor = models.__dict__['resnet50'](num_classes=365)

    def forward(self, features, proposals, targets=None, protect_targets=None, logger=None):
        #time_1 = time.perf_counter()
        self.device=features[0].device
        losses = {}
        self.box.eval()
        with torch.no_grad():
            x, detections, loss_box = self.box(features, proposals, targets, protect_targets)
        
        #time_2 = time.perf_counter()
        #print("roi.box:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        if not self.cfg.MODEL.RELATION_ON and not self.cfg.MODEL.EVENT_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        #if self.cfg.MODEL.ATTRIBUTE_ON:
        if self.cfg.KGEVENT.ATTRIBUTE:
            # Attribute head don't have a separate feature extractor
            detections=self.attribute(features, detections, targets, protect_targets)
            #losses.update(loss_attribute)
        #time_3 = time.perf_counter()

        if self.cfg.MODEL.SCENE_ON:
            scene_info_list=[]
            for protect_target in protect_targets:
                image=protect_target.get_field('image')
                scene_info = self.get_scene_info(image)
                scene_info_list.append(scene_info)
        #time_4 = time.perf_counter()
        #print("roi.scene:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))

        if self.cfg.MODEL.PROTECTED_DETECTION_ON:
            protect_targets=self.protected_area(protect_targets)

        if self.cfg.MODEL.EVENT_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # return x, detections, losses
            #print(detections[0].get_field('pred_labels'),detections[0].get_field('pred_scores'))
            x, detections, loss_event = self.event(features, detections, targets, protect_targets, scene_info_list, logger)
            losses.update(loss_event)
        #time_5 = time.perf_counter()
        
        #print((time_5-time_1)*1000, (time_2-time_1)* 1000, (time_3-time_2)* 1000, (time_4-time_3)* 1000, (time_5-time_4)* 1000)
        '''import csv

        # 假设这些时间变量已经定义
        # time_1, time_2, time_3, time_4, time_5

        # 计算时间差
        time_diffs = [
            (time_5 - time_1) * 1000,
            (time_2 - time_1) * 1000,
            (time_3 - time_2) * 1000,
            (time_4 - time_3) * 1000,
            (time_5 - time_4) * 1000
        ]

        # 指定输出文件路径
        output_file = '/workspace/huangyunyi/res/times.csv'

        # 将时间差写入CSV文件
        with open(output_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(time_diffs)'''

        return x, detections, losses


    def get_scene_info(self, x):
        scene_info = {}
        
        self.scene_recognizor.eval()
        scene_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(45), 
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]), 
        ])
        x = scene_transform(x).to(self.device)
        # print(x)
        with torch.no_grad():
            output = self.scene_recognizor.forward(x.unsqueeze(0))
            # print(output)
        feat_extractor = nn.Sequential(*list(self.scene_recognizor.children())[:-1]).to(self.device)
        feat = feat_extractor.forward(x.unsqueeze(0)).view(-1)
        
        scene_info['feat'] = feat.detach()
        _, indices = torch.sort(output, descending=True)
        scene_info['labels'] = torch.tensor(indices[0][:5])
        scores = torch.nn.functional.softmax(output, dim=1)[0]
        scene_info['scores'] = scores #torch.tensor([scores[idx].item() for idx in indices[0][:5]])
    
        return scene_info


def build_pa_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.EVENT_ON:
        roi_heads.append(("event", build_pa_roi_event_head(cfg, in_channels)))
    if cfg.KGEVENT.ATTRIBUTE:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedPAROIHeads(cfg, roi_heads)

    return roi_heads
