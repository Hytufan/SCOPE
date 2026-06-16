import os
import sys
import json
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dist(A, B):
    squareA = A ** 2
    sum_sq_A = torch.sum(squareA,dim=1).unsqueeze(1)  # m->[m, 1]
    squareB = B ** 2
    sum_sq_B = torch.sum(squareB,dim=1).unsqueeze(0)  # n->[1, n]
    dist = torch.sqrt(sum_sq_A + sum_sq_B - 2 * A.mm(B.t())).to(A.device)
    return dist

def iou(A, B):
    areaA = (A[:,2] - A[:,0]) * (A[:,3] - A[:,1])
    areaB = (B[:,2] - B[:,0]) * (B[:,3] - B[:,1])
    a = areaA.unsqueeze(-1)
    b = areaB.unsqueeze(0)
    union = a + b
    zero = torch.zeros(a.shape[0], b.shape[0], device=A.device)
    a0 = A[:, 0].unsqueeze(-1)
    b0 = B[:, 0].unsqueeze(0)
    a1 = A[:, 1].unsqueeze(-1)
    b1 = B[:, 1].unsqueeze(0)
    a2 = A[:, 2].unsqueeze(-1)
    b2 = B[:, 2].unsqueeze(0)
    a3 = A[:, 3].unsqueeze(-1)
    b3 = B[:, 3].unsqueeze(0)
    i0 = torch.where((a0-b0)>0, a0, b0).to(torch.float)
    i1 = torch.where((a1-b1)>0, a1, b1).to(torch.float)
    i2 = torch.where((a2-b2)<0, a2, b2).to(torch.float)
    i3 = torch.where((a3-b3)<0, a3, b3).to(torch.float)
    iw = torch.where((i2-i0)>0, i2-i0, zero)
    ih = torch.where((i3-i1)>0, i3-i1, zero)
    intersaction = iw * ih
    iou = intersaction / (union - intersaction)
    piou = intersaction / b
    return iou, piou

def theta(A, B):
    theta = float(abs(A[:, 1] - B[:, 1])) / abs(A[:, 0] - B[:, 0])

def vis(A, B):
    # sim =  F.pairwise_distance(A, B, p='cosine')
    # return sim
    sim = A.matmul(B.transpose(-2, -1))
    a = torch.norm(A, p=2, dim=-1)
    b = torch.norm(B, p=2, dim=-1)
    sim /= a.unsqueeze(-1)
    sim /= b.unsqueeze(-2)
    return sim

def obj_scn_prob(matSO, matOS, nodeO, nodeS):
    obj_cls = nodeO['label']
    scn_cls = nodeS['labels']
    scn_sco = nodeS['scores']
    # print("scn cls & sco:", scn_cls, scn_sco)
    p_so, p_os = 0, 0
    for i, c in enumerate(scn_cls):
        # p_so += matSO[obj_cls][c] * scn_sco[i]
        # p_os += matOS[obj_cls][c] * scn_sco[i]
        p_so += matSO[obj_cls][c]
        p_os += matOS[obj_cls][c]
    return p_so, p_os


def save_dict(dict, save_file):
    with open(save_file, 'w') as f:
        json.dump(dict, f)

def save_graph(edges, label, save_file):
    with open(save_file, 'w') as f:
        for src, etype, tgt in edges:
            f.write('{} {} {}\n'.format(src, etype, tgt))
        f.write('? {}\n'.format(label))
        f.write('\n')

#scn_obj_mat = np.load(os.path.join(cfg.root_dir, 'datasets/scene_of_objects.npy'))
#obj_scn_mat = np.load(os.path.join(cfg.root_dir, 'datasets/object_of_scenes.npy'))

class RelationLearner(nn.Module):
    def __init__(self, cfg):
        super(RelationLearner, self).__init__()

        self.device = None
        self.cfg = cfg.clone()
        self.n_nodes = cfg.PROTECTED.GGNN_N_NODES
        self.n_edge_types = cfg.PROTECTED.GGNN_N_EDGE_TYPES

        self.adj_matrix = None
        self.annotation = None

        #self.m_w = nn.Parameter(torch.tensor([0.3]))
        # self.e_vis_w = nn.Parameter(torch.tensor([0.2]))
        # self.e_spa_w = nn.Parameter(torch.tensor([0.8]))
        # self.e_spa_w = nn.Linear(2, 1, device=self.device)
        #self.e_obj_msc_w = nn.Linear(3, 1)
        #self.e_obj_obj_w = nn.Linear(3, 1)
        #self.e_vis_w = nn.Linear(3, 1)
        self.e_vis_spa_alpha = 0.2
        #self.spare = nn.ReLU()

    
    def forward(self, image, m_nodes, o_nodes, s_node):
        # print(m_nodes)
        # print(o_nodes)
        # print(s_node)
        self.device=image.device

        _, h, w = image.shape
        diag_len = torch.sqrt(torch.tensor(float(h)) ** 2 + torch.tensor(float(w)) ** 2).to(self.device)

        self.n_other_objects = len(o_nodes)
        self.n_masked_objects = len(m_nodes)

        o_locs = torch.stack([o['center'] for o in o_nodes])
        m_locs = torch.stack([m['center'] for m in m_nodes])
        o_boxs = torch.stack([o['box'] for o in o_nodes])
        m_boxs = torch.stack([m['box'] for m in m_nodes])
        o_class = torch.stack([o['label'] for o in o_nodes])
        o_feats = torch.stack([o['feat'] for o in o_nodes])
        m_feats = torch.stack([m['feat'] for m in m_nodes])
        s_feat  = s_node['feat']

        node_locs   = torch.cat((m_locs, o_locs), dim=0)
        node_boxs   = torch.cat((m_boxs, o_boxs), dim=0)
        node_feats  = torch.cat((m_feats, o_feats), dim=0)
        node_class  = torch.zeros((self.n_other_objects, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES), device=self.device).scatter_(1, o_class.view(-1,1), 1)

        norm_boxs = node_boxs
        norm_boxs[:, 0] = norm_boxs[:, 0] / w
        norm_boxs[:, 1] = norm_boxs[:, 1] / h
        norm_boxs[:, 2] = norm_boxs[:, 2] / w
        norm_boxs[:, 3] = norm_boxs[:, 3] / h

        o_feat_dim = o_feats.shape[-1]
        m_feat_dim = m_feats.shape[-1]
        s_feat_dim = s_feat.shape[-1]
        assert o_boxs.shape[-1] == m_boxs.shape[-1]
        box_dim = o_boxs.shape[-1]
        cls_dim = self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        self.adj_matrix = torch.zeros([self.n_nodes, self.n_nodes * self.n_edge_types * 2], device=self.device)
        self.annotation = torch.zeros([self.n_nodes, self.cfg.PROTECTED.GGNN_D_ANNOTATION], device=self.device)

        node_vis    = vis(node_feats, node_feats)
        node_dist   = dist(node_locs, node_locs)
        node_iou, node_piou = iou(node_boxs, node_boxs)
        # node_theta  = theta(node_locs, node_locs)
        
        for i in range(self.n_other_objects):
            node_idx = 1 + self.n_masked_objects + i
            # print("object:", node_id)
            self.annotation[node_idx][:o_feat_dim] = o_feats[i]
            self.annotation[node_idx][o_feat_dim:(o_feat_dim + cls_dim)] = node_class[i]
            self.annotation[node_idx][(o_feat_dim + cls_dim): (o_feat_dim + cls_dim + box_dim)] = norm_boxs[self.n_masked_objects + i]
            scene_idx = 0
            self.annotation[scene_idx][:s_feat_dim] = s_feat

            e_type = 1

            #p_so, p_os = obj_scn_prob(scn_obj_mat, obj_scn_mat, o_nodes[i], s_node)
            # src: scene; tgt: object i
            src_idx = scene_idx
            tgt_idx = node_idx
            self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = 1
            self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = 1
            # self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = p_os                            # 2.0
            # self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = p_os        # 2.0
            # src: object i; tgt: scene
            src_idx = node_idx
            tgt_idx = scene_idx
            self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = 1
            self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = 1
            # self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = p_so                            # 2.0
            # self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = p_so        # 2.0

            # object_node : mask_node
            for j in range(self.n_masked_objects):
                msc_idx = 1 + j

                self.annotation[msc_idx][:m_feat_dim] = m_feats[j]
                self.annotation[msc_idx][(m_feat_dim + cls_dim): (m_feat_dim + cls_dim + box_dim)] = norm_boxs[j]

                e_type = 2

                distance = 1.0 / (node_dist[self.n_masked_objects + i, j] / diag_len + 1.0)
                overlap = node_iou[self.n_masked_objects + i, j]
                visual = node_vis[self.n_masked_objects + i, j]
                e_value = 1
                # e_value = self.e_obj_msc_w(torch.tensor([visual, distance, overlap], device=self.device))
                # print("e_obj_msc_w:")
                # for p in self.e_obj_msc_w.parameters():
                #     print(p)

                src_idx = node_idx
                tgt_idx = msc_idx
                self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = e_value
                self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = e_value

            # object_node : object_node
            for k in range(i+1, self.n_other_objects):
                obj_idx = 1 + self.n_masked_objects + k
                
                e_type = 3
                
                distance = 1.0 / (node_dist[self.n_masked_objects + i, self.n_masked_objects + k] / diag_len + 1.0)
                overlap = node_iou[self.n_masked_objects + i, self.n_masked_objects + k]                     # 1.0
                s_overlap = node_piou[self.n_masked_objects + i, self.n_masked_objects + k]
                t_overlap = node_piou[self.n_masked_objects + k, self.n_masked_objects + i]
                # theta = box_theta(o_nodes[i], o_nodes[k])
                visual = node_vis[self.n_masked_objects + i, self.n_masked_objects + k]
                e_value = 1
                # e_value = self.e_obj_obj_w(torch.tensor([visual, distance, overlap], device=self.device))
                # print("e_obj_obj_w:")
                # for p in self.e_obj_obj_w.parameters():
                #     print(p)
                
                src_idx = node_idx
                tgt_idx = obj_idx
                self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = e_value
                self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = e_value
                
                src_idx = obj_idx
                tgt_idx = node_idx
                self.adj_matrix[tgt_idx - 1][(e_type - 1) * self.n_nodes + src_idx - 1] = e_value
                self.adj_matrix[src_idx - 1][(e_type - 1 + self.n_edge_types) * self.n_nodes + tgt_idx - 1] = e_value
        
        # print(self.adj_matrix, self.adj_matrix.shape)
        # print(self.annotation, self.annotation.shape)
        return self.adj_matrix, self.annotation

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
