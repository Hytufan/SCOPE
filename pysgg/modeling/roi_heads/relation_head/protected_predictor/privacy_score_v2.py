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

        self.attr_s=read_score_from_txt('/workspace/huangyunyi/res/attr_sensitivity_personal.txt')
        self.obj_s=read_score_from_txt('/workspace/huangyunyi/res/obj_sensitivity_personal_test.txt')
        self.event_s=read_score_from_txt('/workspace/huangyunyi/res/event_sensitivity_personal.txt')
        self.scene_s=read_score_from_txt('/workspace/huangyunyi/res/scene_sensitivity_personal.txt')
        #self.obj_s=read_score_from_txt('/workspace/huangyunyi/data/PrivacyAlert2/object_sensitivity_my.txt')
        #self.event_s=read_score_from_txt('/workspace/huangyunyi/data/PrivacyAlert2/event_sensitivity_my.txt')
        #self.o_e_s=read_score_from_npy('/workspace/huangyunyi/res/o_e_sensitivity.npy')
        
        self.person_labels=[20,29,53,56,68,70,78,79,98,119,149]
        self.k_a,self.k_e,self.k_e_p,self.k_oe= [0.2, 3, 1.0, 0.1]
        self.filter=False

    def forward(self, proposal, eve_cls_logits, msk_cls_logits, mask_adj):
        
        pred_labels = proposal.get_field('pred_labels')
        label_scores = proposal.get_field('pred_scores')
        attributes = proposal.get_field('attribute')
        scene =  proposal.get_field('pred_scene')
        
        if self.filter:
            label_scores[label_scores < 0.7] = 0
        
        if mask_adj==None:
            mask_adj=wrap(np.ones(pred_labels.shape))
        
        #print(filename)
        obj_sen_scores = self.obj_s[pred_labels]
        
        if self.attr_on:

            for key, value in attributes.items():
                max_race_score = value['race_score'].max()
                max_gender_score = value['gender_score'].max()
                max_age_score = value['age_score'].max()
                
                if self.filter:
                    max_race_score = max_race_score if max_race_score >= 0.7 else 0
                    max_gender_score = max_gender_score if max_gender_score >= 0.7 else 0
                    max_age_score = max_age_score if max_age_score >= 0.7 else 0
                
                attr_score = max_age_score * self.attr_s[0] + max_gender_score * self.attr_s[1] + max_race_score * self.attr_s[2]
                #mask_adj_value = mask_adj[key]
                #attr_score = attr_score + attr_sum * mask_adj_value
                obj_sen_scores[key]=obj_sen_scores[key] + self.k_a * attr_score
                #print(obj_sen_scores[key])
        #print(attr_score,len(attributes))
        
        '''event_label=torch.argmax(eve_cls_logits, dim=-1)
        if F.softmax(eve_cls_logits, dim=1).max()>0.7:
            event_score=(self.event_s[event_label.item()] * F.softmax(eve_cls_logits, dim=1).max())
        else:
            event_score=0'''
            
        event_probs = F.softmax(eve_cls_logits, dim=1)
        event_conf = event_probs.max()
        event_label = torch.argmax(event_probs, dim=-1)
        if self.filter and event_conf < 0.7:
            event_conf=0
        event_score = self.event_s[event_label.item()] * event_conf
        
        for i in range(len(pred_labels.tolist())):
            label=pred_labels.tolist()[i]
            if label in self.person_labels:
                obj_sen_scores[i]=obj_sen_scores[i] + self.k_e * self.k_e_p * event_score
            if label==44:
                obj_sen_scores[i]=obj_sen_scores[i] + self.k_e * (1-self.k_e_p) * event_score
        
        obj_scores = mask_adj * obj_sen_scores * label_scores
        obj_score = obj_scores.sum()
        
        scene_score=self.scene_s[scene.item()]
        
        private_score = obj_score + self.k_oe * event_score

        return private_score, event_score