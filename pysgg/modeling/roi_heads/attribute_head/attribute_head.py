# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from .roi_attribute_predictors import make_roi_attribute_predictor
from .loss import make_roi_attribute_loss_evaluator
import numpy as np
import dlib
from torchvision import datasets, models, transforms
import torchvision
from pysgg.structures.bounding_box import BoxList
import torch.nn.functional as F

def add_attribute_logits(proposals, attri_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("attribute_logits", attri_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

class ROIAttributeHead(torch.nn.Module):
    """
    Generic ATTRIBUTE Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIAttributeHead, self).__init__()
        self.cfg = cfg.clone()
        #self.feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        #self.predictor = make_roi_attribute_predictor(cfg, self.feature_extractor.out_channels)
        #self.loss_evaluator = make_roi_attribute_loss_evaluator(cfg)
        self.model_fair_7 = torchvision.models.resnet34(pretrained=True)
        self.model_fair_7.fc = nn.Linear(self.model_fair_7.fc.in_features, 18)
        self.model_fair_7.load_state_dict(torch.load('/workspace/huangyunyi/code/PySGG/pysgg/modeling/roi_heads/attribute_head/ckpt/res34_fair_align_multi_7_20190809.pt'))
        self.model_fair_7 = self.model_fair_7

        self.trans = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, features, proposals, targets=None, protect_targets=None):

        to_pil = transforms.ToPILImage()

        for j, proposal, protect_target in zip(range(len(proposals)),proposals,protect_targets):
            attribute={}

            obj_targets=BoxList(proposal.bbox,proposal.size,proposal.mode)
            image=to_pil(protect_target.get_field('image'))
            width, height = image.size
            obj_targets=obj_targets.resize((width, height))
            device=features[0].device

            labels = proposal.get_field('pred_labels').tolist()
            for i in range(len(labels)):
                label=labels[i]
                if label==44:
                    bbox=obj_targets.bbox[i].cpu().tolist()
                    face_image=image.crop(bbox)
            
                    face_image = self.trans(face_image)
                    face_image = face_image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
                    face_image = face_image.to(device)

                    # fair
                    self.model_fair_7.eval()
                    outputs = self.model_fair_7(face_image).squeeze()
                    #outputs = outputs.cpu().detach().numpy()
                    #outputs = np.squeeze(outputs)

                    race_outputs = outputs[:7]
                    gender_outputs = outputs[7:9]
                    age_outputs = outputs[9:18]
                    #print(race_outputs)
                    race_score = F.softmax(race_outputs,dim=0)
                    gender_score = F.softmax(gender_outputs,dim=0)
                    age_score = F.softmax(age_outputs,dim=0)

                    # 获取每个类别的预测结果
                    race_pred = torch.argmax(race_score).item()
                    gender_pred = torch.argmax(gender_score).item()
                    age_pred = torch.argmax(age_score).item()
                    #print(race_score,race_pred,gender_score,gender_pred,age_score,age_pred)

                    person_attr={}
                    person_attr['race_score']=race_score
                    person_attr['race_pred']=race_pred
                    person_attr['gender_score']=gender_score
                    person_attr['gender_pred']=gender_pred
                    person_attr['age_score']=race_score
                    person_attr['age_pred']=age_pred

                    attribute[i]=person_attr
            proposals[j].add_field('attribute',attribute)

        return proposals

def build_roi_attribute_head(cfg, in_channels):
    """
    Constructs a new attribute head.
    By default, uses ROIAttributeHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIAttributeHead(cfg, in_channels)




