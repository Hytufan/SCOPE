# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from pysgg.structures.image_list import to_image_list
from pysgg.modeling.make_layers import make_fc
from .generalized_rcnn import GeneralizedRCNN
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier

class GraphReadout(nn.Module):
    def __init__(self):
        super(GraphReadout, self).__init__()

    def forward(self, x, mask=None):
        graph_representation = x.mean(dim=-2)
        return graph_representation



class EventNet(nn.Module):
    def __init__(self, cfg):
        super(EventNet, self).__init__()
        self.cfg = cfg.clone()
        self.relationNet=GeneralizedRCNN(cfg)
        self.readout = GraphReadout()
        if cfg.EVENT.FROM_NODE_TYPE=='obj':
            self.fc1=nn.Sequential(
                        make_fc(cfg.PROTECTED.GGNN_D_STATE, cfg.PROTECTED.GGNN_D_STATE // 2),
                        nn.ReLU(True),
                    )
        elif cfg.EVENT.FROM_NODE_TYPE=='objrel':
            self.fc1=nn.Sequential(
                        make_fc(cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM, cfg.PROTECTED.GGNN_D_STATE // 2),
                        nn.ReLU(True),
                    )
            self.fc2=nn.Sequential(
                        make_fc(cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM, cfg.PROTECTED.GGNN_D_STATE // 2),
                        nn.ReLU(True),
                    )
            self.fc3=nn.Sequential(
                        make_fc(cfg.PROTECTED.GGNN_D_STATE // 2, cfg.PROTECTED.GGNN_D_STATE // 2),
                        nn.ReLU(True),
                    )
        '''self.fc4=nn.Sequential(
                        make_fc(cfg.PROTECTED.GGNN_D_STATE // 2, cfg.PROTECTED.GGNN_D_STATE // 2),
                        nn.ReLU(True),
                    )'''
        #self.fc2=torch.nn.Linear(cfg.PROTECTED.GGNN_D_STATE // 2, self.cfg.EVENT.NUM_CLASSES)
        self.classifier=build_classifier(cfg.PROTECTED.GGNN_D_STATE // 2, self.cfg.EVENT.NUM_CLASSES)
        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, images, targets=None, protect_targets=None, logger=None):
        batch_size=len(protect_targets)

        self.relationNet.eval()
        with torch.no_grad():
            nodes_output, result, bgnn_feats=self.relationNet(images,targets,protect_targets,logger)
            rel_feats=bgnn_feats["rel_feats"]
            obj_feats=bgnn_feats["obj_feats"]

        pred_conf_all=None
        for i in range(len(nodes_output)):
            if self.cfg.EVENT.FROM_NODE_TYPE=='obj':
                node_output = nodes_output[i]
                x = self.fc1(node_output)
                x = self.readout(x)
                x=x.unsqueeze(0)
                #x=self.fc4(x)
            elif self.cfg.EVENT.FROM_NODE_TYPE=='objrel':
                rel_feat=self.fc1(rel_feats[i])
                obj_feat=self.fc2(obj_feats[i])
                rel_feat=self.readout(rel_feat)
                obj_feat=self.readout(obj_feat)
                x=rel_feat+obj_feat
                x=x.unsqueeze(0)
                x = self.fc3(x)
            pred_conf=self.classifier(x)
            pred_label = torch.argmax(pred_conf, dim=-1)
            result[i].add_field("pred_event_label",pred_label)
            target_label=protect_targets[i].get_field("event_label")

            if pred_conf_all==None:
                pred_conf_all=pred_conf
                target_labels_all=target_label
            else:
                pred_conf_all=torch.cat((pred_conf_all,pred_conf))
                target_labels_all=torch.cat((target_labels_all,target_label))
        if self.training:
            loss = self.criterion_loss(pred_conf_all, target_labels_all.long())

            output_losses = dict(loss_event=loss)
            output_losses_checked = {}
            for key in output_losses.keys():
                if output_losses[key] is not None:
                    if output_losses[key].grad_fn is not None:
                        output_losses_checked[key] = output_losses[key]
            output_losses = output_losses_checked
            return output_losses
        else:
            return result
            
