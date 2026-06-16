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

from .ggnn_kg_scene import GGNN
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


class GGNNRelReason(nn.Module):
    """
    Module for relationship classification.
    """
    def __init__(self, cfg, obj_dim=4096, event_dim=4096, 
                time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, use_embedding=True, 
                 top_k_to_keep=5, normalize_messages=True):

        super(GGNNRelReason, self).__init__()
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
        self.obj_and_event=True


        self.obj_proj = nn.Linear(self.obj_dim, self.hidden_dim)
        self.rel_proj = nn.Linear(self.event_dim, self.hidden_dim)
        
        self.ggnn = GGNN(cfg=cfg,time_step_num=time_step_num, hidden_dim=self.hidden_dim, output_dim=output_dim, 
                         emb_path=self.emb_path, graph_path=self.graph_path,
                         use_knowledge=use_knowledge, use_embedding=use_embedding, top_k_to_keep=top_k_to_keep, 
                         normalize_messages=normalize_messages)
        self.scene_recognizor = models.__dict__['resnet50'](num_classes=365)

        #self.smooth_loss=LabelSmoothingLoss(classes=self.num_event_cls)
        '''self.OntReader = OntologyReader(graph_file='/workspace/huangyunyi/data/WIDER_v0.1/ontology/ontology_WIDER.graphml',
                               weighting_scheme='distance',
                               leaf_node_weight=1.0)'''
        self.loss_weight=np.load('/workspace/huangyunyi/data/WIDER_v0.1/event_weight_log2.npy')
        self.loss_weight=torch.from_numpy(self.loss_weight).float()
        self.criterion_loss = nn.CrossEntropyLoss(weight=self.loss_weight)
        self.criterion_loss_obj = nn.CrossEntropyLoss()

    def forward(self, proposals, protect_targets,roi_features):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        """
        #obj_fmaps, obj_logits, event_features
        self.device=roi_features.device
        
        event_logits = []
        obj_logits_refined = []
        idx=0
        pred_conf_all=None
        pred_obj_all=None
        pred_conf_all_list=[]
        losses=[]
        for proposal, protect_target in zip(proposals,protect_targets):
            roi_feature=roi_features[idx:idx+len(proposal),:]
            idx+=len(proposal)
            image=protect_target.get_field('image')
            scene_info = self.get_scene_info(image)

            el, ol = self.ggnn(proposal, roi_feature, scene_info, image)
            event_logits.append(el)
            obj_logits_refined.append(ol)

            pred_label = torch.argmax(el, dim=-1)
            proposal.add_field("pred_event_label",pred_label)

            if self.obj_and_event:
                pred_obj = torch.argmax(ol, dim=-1)
                proposal.add_field("pred_obj_label",torch.tensor([self.class_map[pred_obj]]))

            target_label=protect_target.get_field("event_label")

            '''leaf_node_vector = self.OntReader.subgraph_to_leaf_vector(pred_subgraph_vector=el,
                                                                     strategy='cossim',
                                                                     redundancy_removal=False)
            print(leaf_node_vector)'''

            #el_conf[0,int(target_label.item())]=1
            #el=torch.mul(el, el_conf)
            
            if pred_conf_all==None:
                pred_conf_all=el
                target_labels_all=target_label
                #el_conf_all=el_conf
                '''for i in range(len(el_steps)):
                    pred_conf_all_list.append(el_steps[i])'''
                if self.obj_and_event:
                    pred_obj_all=ol
                    target_obj=protect_target.get_field("protect_labels")
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
                    target_obj_trans=torch.zeros_like(target_obj, dtype=torch.long)
                    for i in range(len(target_obj)):
                        target_obj_trans[i] = list(self.class_map).index(target_obj[i])
                    target_obj_all=torch.cat((target_obj_all,target_obj_trans))
            
        loss = self.criterion_loss(pred_conf_all, target_labels_all.long())
        #if pred_obj_all != None:
        if self.obj_and_event:
            loss_obj=self.criterion_loss_obj(pred_obj_all, target_obj_all.long())
            loss=[loss,loss_obj]
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
        
        # get scene package:
        # idx = str(int(idx)).rjust(12, '0')
        # scene_preds = pickle.load( \
        #         open(os.path.join(dataset_dir, 'scenes', idx + ".pkl"), 'rb'))
        # scene_info['feat'] = scene_preds['feat'].detach()
        # scene_info['labels'] = scene_preds['labels']
        # scene_info['scores'] = np.array(scene_preds['scores'])
    
        return scene_info


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
        return F.cross_entropy(result.rel_dists, result.rel_labels[:, -1], weight=self.rel_class_weights)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.2, dim=-1):

        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # 置信度，即非平滑部分的权重
        self.smoothing = smoothing  # 平滑系数
        self.cls = classes  # 类别总数
        self.dim = dim  # softmax操作的维度

    def forward(self, pred, target, weight):

        row_sums = weight.sum(dim=1, keepdim=True)
        normalized_weight = weight / row_sums
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            #true_dist = torch.zeros_like(pred) 
            true_dist=normalized_weight*self.smoothing
            true_dist[torch.arange(true_dist.size(0)), target] += self.confidence
        #print(true_dist)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

