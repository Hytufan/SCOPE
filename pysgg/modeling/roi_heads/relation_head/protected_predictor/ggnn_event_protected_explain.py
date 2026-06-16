"""
from my_model_32: new ggnn
"""
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch_geometric.explain import GNNExplainer, Explainer

from .ggnn_pa import GGNN
from .relation import RelationLearner
import heapq
'''from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, onehot_logits, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.surgery import filter_dets
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.my_ggnn_16 import GGNN'''

np.set_printoptions(threshold=sys.maxsize)

#MODES = ('sgdet', 'sgcls', 'predcls')

import datetime


class GGNNEventReason(nn.Module):
    """
    Module for relationship classification.
    """
    def __init__(self, cfg, obj_dim=4096, event_dim=4096, 
                time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=False, use_embedding=True, 
                 top_k_to_keep=5, normalize_messages=True):

        super(GGNNEventReason, self).__init__()
        self.cfg=cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.graph_path=cfg.KGEVENT.GRAPH_PATH
        self.emb_path=cfg.KGEVENT.EMB_PATH
        self.num_event_cls = cfg.EVENT.NUM_CLASSES
        self.obj_dim = obj_dim
        self.event_dim = event_dim
        self.device=None
        self.hidden_dim=cfg.KGEVENT.HIDDEN_DIM
        self.class_map=cfg.PROTECTED.PROTECT_CLASSES
        if len(self.class_map)==0:
            self.class_map=list(range(151))
        self.obj_and_event=True


        self.obj_proj = nn.Linear(self.obj_dim, self.hidden_dim)
        self.eve_proj = nn.Linear(self.event_dim, self.hidden_dim)
        
        self.ggnn = GGNN(cfg=cfg,time_step_num=time_step_num, hidden_dim=self.hidden_dim, output_dim=output_dim, 
                         emb_path=self.emb_path, graph_path=self.graph_path,
                         use_knowledge=cfg.KGEVENT.USE_KG, top_k_to_keep=top_k_to_keep, 
                         normalize_messages=normalize_messages)
        self.relation_learner = RelationLearner(cfg)

        #self.smooth_loss=LabelSmoothingLoss(classes=self.num_event_cls)
        '''self.OntReader = OntologyReader(graph_file='/workspace/huangyunyi/data/WIDER_v0.1/ontology/ontology_WIDER.graphml',
                               weighting_scheme='distance',
                               leaf_node_weight=1.0)'''
        #self.loss_weight=np.load('/workspace/huangyunyi/data/WIDER_v0.1/event_weight_log2.npy')
        #self.loss_weight=torch.from_numpy(self.loss_weight).float()
        #self.criterion_loss = nn.CrossEntropyLoss(weight=self.loss_weight)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.criterion_loss_obj = nn.CrossEntropyLoss()

        self.explainer = Explainer(
            model=self.ggnn,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='phenomenon',
            node_mask_type='object',
            edge_mask_type=None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
        )

    def forward(self, proposals, protect_targets,roi_features,roi_features_protected,scene_info_list):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        """
        #obj_fmaps, obj_logits, event_features
        self.device=roi_features.device
        
        event_logits = []
        obj_logits_refined = []
        idx=0
        idx_p=0
        pred_conf_all=None
        pred_obj_all=None
        pred_conf_all_list=[]
        losses=[]
        #print("event0:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        for proposal, protect_target, scene_info in zip(proposals,protect_targets,scene_info_list):
            roi_feature=roi_features[idx:idx+len(proposal),:]
            roi_feature_protected=roi_features_protected[idx_p:idx_p+len(protect_target),:]
            idx+=len(proposal)
            idx_p+=len(protect_target)

            image=protect_target.get_field('image')
            masked_info, others_info = self.get_objects_info(image, proposal, protect_target, roi_feature, roi_feature_protected)
            adj_matrix, annotation = self.relation_learner(self.device, image, masked_info, others_info)
            #print("info:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
            #el, ol = self.ggnn(proposal, roi_feature, scene_info)
            explanation = self.explainer(annotation, None, target=protect_target.get_field("event_label"), proposal=proposal, scene_info=scene_info, adj_matrix=adj_matrix, image=image)
            #print(explanation['node_mask'].squeeze(dim=1),explanation['node_mask'].squeeze(dim=1).shape)
            #print(proposal.get_field('pred_labels'),proposal.get_field('pred_labels').shape)
            proposal.add_field("explain",explanation['node_mask'].squeeze(dim=1))
            proposal.add_field("mask_bbox",protect_target.bbox)

            #print("ggnn:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
            
            el, ol, ops, eps = self.ggnn(annotation, proposal, scene_info, adj_matrix, image)
            event_logits.append(el)
            obj_logits_refined.append(ol)

            proposal.add_field("el",el)
            proposal.add_field("ol",ol)
            proposal.add_field("ops",ops)
            proposal.add_field("eps",eps)

            pred_label = torch.argmax(el, dim=-1)
            proposal.add_field("pred_event_label",pred_label)

            if self.obj_and_event:
                pred_obj = torch.argmax(ol, dim=-1)
                pred_obj_list=pred_obj.tolist()
                pred_obj_labels=[self.class_map[pred_obj_index] for pred_obj_index in pred_obj_list]
                #proposal.add_field("pred_obj_label",torch.tensor([self.class_map[pred_obj]]))
                proposal.add_field("pred_obj_label",torch.tensor(pred_obj_labels))
            else:
                proposal.add_field("pred_obj_label",torch.tensor([0]))

            target_label=protect_target.get_field("event_label")
            proposal.add_field("event_label",target_label)
            
            if pred_conf_all==None:
                pred_conf_all=el
                target_labels_all=target_label
                #el_conf_all=el_conf
                '''for i in range(len(el_steps)):
                    pred_conf_all_list.append(el_steps[i])'''
                if self.obj_and_event:
                    pred_obj_all=ol
                    target_obj=protect_target.get_field("protect_labels")
                    proposal.add_field("protect_labels",target_obj)
                    target_obj_trans=torch.zeros_like(target_obj, dtype=torch.long)
                    for i in range(len(target_obj)):
                        target_obj_trans[i] = list(self.class_map).index(target_obj[i])
                    target_obj_all=target_obj_trans
            else:
                pred_conf_all=torch.cat((pred_conf_all,el))
                target_labels_all=torch.cat((target_labels_all,target_label))
                #el_conf_all=torch.cat((el_conf_all,el_conf))
                '''for i in range(len(el_steps)):
                    pred_conf_all_list[i]=torch.cat((pred_conf_all_list[i],el_steps[i]))'''
                if self.obj_and_event:
                    pred_obj_all=torch.cat((pred_obj_all,ol))
                    target_obj=protect_target.get_field("protect_labels")
                    proposal.add_field("protect_labels",target_obj)
                    target_obj_trans=torch.zeros_like(target_obj, dtype=torch.long)
                    for i in range(len(target_obj)):
                        target_obj_trans[i] = list(self.class_map).index(target_obj[i])
                    target_obj_all=torch.cat((target_obj_all,target_obj_trans))
            #print("event1:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        #print("event2:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        loss = self.criterion_loss(pred_conf_all, target_labels_all.long())
        #if pred_obj_all != None:
        loss=[loss]
        if self.obj_and_event:
            loss_obj=self.criterion_loss_obj(pred_obj_all, target_obj_all.long())
            loss.append(loss_obj)
        '''for i in range(len(loss)):
            if loss[i].isnan(): 
                loss[i]=1e-6
            else: 
                loss[i] = loss[i].item()
        print(loss)'''
        '''for i in range(len(el_steps)):
            losses.append(self.criterion_loss(pred_conf_all_list[i], target_labels_all.long()))'''
        #print(loss,losses)
        #loss=self.smooth_loss(pred_conf_all, target_labels_all.long(),el_conf_all)
        #print(loss)
            
        #event_logits = torch.cat(event_logits, 0)
        
        if self.ggnn.refine_obj_cls:
            obj_logits_refined = torch.cat(obj_logits_refined, 0)
            obj_logits = obj_logits_refined

        return proposals, loss, obj_logits

    def get_objects_info(self, image, proposal, protect_target, roi_feature, roi_feature_protected):
        masked_info = []
        c, h, w = image.shape

        masked_boxes = protect_target.bbox
        for i in range(len(masked_boxes)):
            masked = {}
            masked['box'] = masked_boxes[i]
            x1, y1, x2, y2 = masked_boxes[i]
            masked['center'] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2]).to(self.device)
            masked['area'] = torch.tensor((x2 - x1) * (y2 - y1)).to(self.device)
            masked_info.append(masked)

        n_masked_objects = len(masked_info)
        others_info = []
        n_other_objects = len(proposal)
        
        for i in range(n_other_objects):
            obj = {}
            obj['box'] = proposal.bbox[i]
            x1, y1, x2, y2 = proposal.bbox[i]
            obj['score'] = proposal.get_field('pred_scores')[i]
            obj['label'] = proposal.get_field('pred_labels')[i]
            obj['feat'] = roi_feature[i]
            obj['center'] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2]).to(self.device)
            obj['area'] = torch.tensor((x2 - x1) * (y2 - y1)).to(self.device)
            others_info.append(obj)
            
        for i in range(n_masked_objects):
            masked_info[i]['feat'] = roi_feature_protected[i]

        return masked_info, others_info

'''
class KERN(nn.Module):
    """
    Knowledge-Embedded Routing Network 
    """
    def __init__(self, cfg,
                 ggnn_rel_time_step_num=3,
                 ggnn_rel_hidden_dim=512,
                 ggnn_rel_output_dim=512, use_knowledge=True, use_embedding=True,
                 top_k_to_keep=5, normalize_messages=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(KERN, self).__init__()
        self.cfg=cfg
        self.classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.graph_path=cfg.KGEVENT.GRAPH_PATH
        self.emb_path=cfg.KGEVENT.EMB_PATH
        self.event_classes = cfg.EVENT.NUM_CLASSES

        self.ggnn_rel_reason = GGNNRelReason(cfg=cfg,
                                             obj_dim=self.obj_dim, 
                                             time_step_num=ggnn_rel_time_step_num, 
                                             hidden_dim=ggnn_rel_hidden_dim, 
                                             output_dim=ggnn_rel_output_dim,
                                             use_knowledge=use_knowledge, 
                                             use_embedding=use_embedding,
                                             top_k_to_keep=top_k_to_keep,
                                             normalize_messages=normalize_messages
                                            )

    def forward(self, proposals,protect_targets,roi_features):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """

        obj_logits,event_logits = self.ggnn_rel_reason(proposals,protect_targets,roi_features)   

        return obj_logits,event_logits

    def obj_loss(self, result):
        if self.ggnn_rel_reason.ggnn.refine_obj_cls:
            return F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        else:
            return Variable(torch.from_numpy(np.zeros((1))).float().cuda(), requires_grad=False)

    def rel_loss(self, result):
        return F.cross_entropy(result.rel_dists, result.rel_labels[:, -1], weight=self.rel_class_weights)'''


