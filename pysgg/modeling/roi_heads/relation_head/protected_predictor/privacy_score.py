import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from torch.nn import init
import pandas as pd
import torchvision.transforms as transforms

def wrap(nparr):
    return Variable(torch.from_numpy(nparr).float().cuda(), requires_grad=False)

def read_score_from_txt(file_path):
    nparr = np.loadtxt(file_path)
    tensor = wrap(nparr)
    return tensor

def read_score_from_npy(file_path):
    nparr = np.load(file_path)
    tensor = wrap(nparr)
    return tensor

class PrivacyScore(torch.nn.Module):

    def __init__(self, cfg):
        super(PrivacyScore, self).__init__()
        self.cfg = cfg.clone()
        self.attr_on=cfg.KGEVENT.ATTRIBUTE

        self.attr_s=read_score_from_txt('/workspace/huangyunyi/res/attribute_sensitivity.txt')
        self.obj_s=read_score_from_txt('/workspace/huangyunyi/res/object_sensitivity.txt')
        self.event_s=read_score_from_txt('/workspace/huangyunyi/res/event_sensitivity.txt')
        #self.obj_s=read_score_from_txt('/workspace/huangyunyi/data/PrivacyAlert2/object_sensitivity_my.txt')
        #self.event_s=read_score_from_txt('/workspace/huangyunyi/data/PrivacyAlert2/event_sensitivity_my.txt')
        #self.o_e_s=read_score_from_npy('/workspace/huangyunyi/res/o_e_sensitivity.npy')

    def forward(self, proposal, eve_cls_logits, msk_cls_logits, mask_adj):
        
        labels=proposal.get_field('pred_labels')
        label_scores=proposal.get_field('pred_scores')
        if self.attr_on:
            attributes=proposal.get_field('attribute')

        if mask_adj==None:
            mask_adj=wrap(np.ones(labels.shape))

        obj_sen_scores = self.obj_s[labels]

        #selected_obj_s = self.obj_s[labels]
        #obj_scores = selected_obj_s * label_scores
        #obj_scores = mask_adj * obj_scores
        #obj_score = obj_scores.sum()
        #print(obj_score,labels.shape)

        if self.attr_on:
            #attr_sum=0
            #attr_score=torch.tensor(0).float().cuda()

            for key, value in attributes.items():
                max_race_score = value['race_score'].max()
                max_gender_score = value['gender_score'].max()
                max_age_score = value['age_score'].max()
                attr_score = max_age_score * self.attr_s[0] + max_gender_score * self.attr_s[1] + max_race_score * self.attr_s[2]
                #mask_adj_value = mask_adj[key]
                #attr_score = attr_score + attr_sum * mask_adj_value
                obj_sen_scores[key]=obj_sen_scores[key]+attr_score
                #print(obj_sen_scores[key])
            #print(attr_score,len(attributes))
                
        obj_scores = mask_adj * obj_sen_scores * label_scores
        obj_score = obj_scores.sum()

        event_label=torch.argmax(eve_cls_logits, dim=-1)
        event_score=(self.event_s[event_label.item()] + obj_score) * F.softmax(eve_cls_logits, dim=1).max()
        #print(event_score,event_label)

        '''selected_o_e_s = self.o_e_s[event_label].squeeze()
        selected_oe_scores = selected_o_e_s[labels]
        oe_scores = selected_oe_scores * label_scores
        oe_scores = mask_adj * oe_scores
        oe_score = oe_scores.sum()'''
        #print(oe_score,len(oe_scores))

        '''if self.attr_on:
            ops=obj_score+attr_score
            eps=event_score+oe_score
        else:
            ops=obj_score
            eps=event_score+oe_score'''
        #print(event_score)

        return event_score, obj_scores