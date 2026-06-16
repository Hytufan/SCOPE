import os
import sys
import json
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import datetime

sys.path.append('..')

def dist0(A, B):
    squareA = A ** 2
    sum_sq_A = torch.sum(squareA,dim=1).unsqueeze(1)  # m->[m, 1]
    squareB = B ** 2
    sum_sq_B = torch.sum(squareB,dim=1).unsqueeze(0)  # n->[1, n]
    dist = torch.sqrt(sum_sq_A + sum_sq_B - 2 * A.mm(B.t())).to(A.device)
    return dist

def dist(A, B):
    squareA = A ** 2
    sum_sq_A = torch.sum(squareA, dim=1, keepdim=True)  # m->[m, 1]
    
    squareB = B ** 2
    sum_sq_B = torch.sum(squareB, dim=1, keepdim=True)  # n->[n, 1]
    
    cross_term = A.mm(B.t())  # [m, n]
    
    d = torch.sqrt(torch.clamp(sum_sq_A + sum_sq_B - 2 * cross_term, min=1e-8))
    
    return d.to(A.device)


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

        self.cfg=cfg
        self.n_nodes = self.cfg.PROTECTED.GGNN_N_NODES
        #self.n_edge_types = self.cfg.PROTECTED.GGNN_N_EDGE_TYPES

        #self.m_w = nn.Parameter(torch.tensor([0.3]))
        # self.e_vis_w = nn.Parameter(torch.tensor([0.2]))
        # self.e_spa_w = nn.Parameter(torch.tensor([0.8]))
        # self.e_spa_w = nn.Linear(2, 1, device=self.device)
        self.e_obj_msc_w = nn.Linear(3, 1)
        self.e_obj_obj_w = nn.Linear(3, 1)
        self.e_msc_msc_w = nn.Linear(3, 1)
        #self.e_vis_w = nn.Linear(3, 1)
        #self.e_vis_spa_alpha = 0.2
        #self.spare = nn.ReLU()

        if cfg.PROTECTED.USE_WORD_EMBEDDING:

            df = pd.read_csv('/workspace/huangyunyi/data/WIDER_v0.1/KG/obj.csv')
            embedding_columns = df.columns[1:]
            embedding_data = df[embedding_columns]
            self.emb_obj = embedding_data.to_numpy()
            zero_row = np.zeros((1, df.shape[1]-1))
            self.emb_obj = torch.from_numpy(np.vstack((zero_row, self.emb_obj)))

            df = pd.read_csv('/workspace/huangyunyi/data/WIDER_v0.1/KG/scene.csv')
            embedding_columns = df.columns[1:]
            embedding_data = df[embedding_columns]
            self.emb_sce = torch.from_numpy(embedding_data.to_numpy())
        
        self._initialize_weights()

    
    def forward(self, device, image, m_nodes, o_nodes):

        self.device=device
        _, h, w = image.shape
        diag_len = torch.sqrt(torch.tensor(float(h)) ** 2 + torch.tensor(float(w)) ** 2).to(self.device)

        self.n_other_objects = len(o_nodes)
        self.n_masked_objects = len(m_nodes)
        self.n_nodes=self.n_other_objects+self.n_masked_objects

        o_locs = torch.stack([o['center'] for o in o_nodes])
        m_locs = torch.stack([m['center'] for m in m_nodes])
        o_boxs = torch.stack([o['box'] for o in o_nodes])
        m_boxs = torch.stack([m['box'] for m in m_nodes])
        o_class = torch.stack([o['label'] for o in o_nodes])
        o_feats = torch.stack([o['feat'] for o in o_nodes])
        m_feats = torch.stack([m['feat'] for m in m_nodes])
        #s_feat  = s_node['feat']

        node_locs   = torch.cat((m_locs, o_locs), dim=0)
        node_boxs   = torch.cat((m_boxs, o_boxs), dim=0)
        node_feats  = torch.cat((m_feats, o_feats), dim=0)
        if self.cfg.PROTECTED.USE_WORD_EMBEDDING:
            node_class = self.emb_obj[o_class].to(device)
        else:
            node_class = torch.zeros((self.n_other_objects, self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES), device=self.device).scatter_(1, o_class.view(-1,1), 1)


        norm_boxs = node_boxs
        norm_boxs[:, 0] = norm_boxs[:, 0] / w
        norm_boxs[:, 1] = norm_boxs[:, 1] / h
        norm_boxs[:, 2] = norm_boxs[:, 2] / w
        norm_boxs[:, 3] = norm_boxs[:, 3] / h

        o_feat_dim = o_feats.shape[-1]
        m_feat_dim = m_feats.shape[-1]
        #s_feat_dim = s_feat.shape[-1]
        assert o_boxs.shape[-1] == m_boxs.shape[-1]
        box_dim = o_boxs.shape[-1]
        if self.cfg.PROTECTED.USE_WORD_EMBEDDING:
            cls_dim = self.emb_obj.size(-1)
        else:
            cls_dim = self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        #self.adj_matrix = torch.zeros([self.n_nodes, self.n_nodes * self.n_edge_types * 2], device=self.device)
        adj_ent2ent=torch.zeros([self.n_nodes, self.n_nodes], device=self.device) #[pro+obj,pro+obj]
        annotation = torch.zeros([self.n_nodes, self.cfg.PROTECTED.GGNN_D_ANNOTATION], device=self.device)

        node_vis    = vis(node_feats, node_feats)
        node_dist   = dist(node_locs, node_locs)
        node_iou, node_piou = iou(node_boxs, node_boxs)
        node_dist = 1.0 / (node_dist / diag_len + 1.0)
        # node_theta  = theta(node_locs, node_locs)

        for i in range(self.n_masked_objects):
            msc_idx = i

            annotation[msc_idx][:m_feat_dim] = m_feats[i]
            annotation[msc_idx][(m_feat_dim + cls_dim): (m_feat_dim + cls_dim + box_dim)] = norm_boxs[i]
        
        for i in range(self.n_other_objects):
            node_idx = self.n_masked_objects + i
            # print("object:", node_id)
            annotation[node_idx][:o_feat_dim] = o_feats[i]
            annotation[node_idx][o_feat_dim:(o_feat_dim + cls_dim)] = node_class[i]
            annotation[node_idx][(o_feat_dim + cls_dim): (o_feat_dim + cls_dim + box_dim)] = norm_boxs[self.n_masked_objects + i]

        for i in range(self.n_nodes-1):
            adj_ent2ent[i][i]=1

        return adj_ent2ent, annotation
        
        edge_feat=torch.stack([node_vis, node_dist, node_iou], dim=-1)
        #edge_feat=edge_feat.view(-1, 3)

        edge_feat_msc_msc=edge_feat[:self.n_masked_objects,:self.n_masked_objects]
        edge_feat_msc_msc=edge_feat_msc_msc.reshape(-1,3)
        edge_feat_msc_msc=self.e_msc_msc_w(edge_feat_msc_msc)
        edge_feat_msc_msc=edge_feat_msc_msc.reshape(self.n_masked_objects,self.n_masked_objects)
        adj_ent2ent[:self.n_masked_objects,:self.n_masked_objects]=edge_feat_msc_msc

        edge_feat_obj_obj=edge_feat[self.n_masked_objects:,self.n_masked_objects:]
        edge_feat_obj_obj=edge_feat_obj_obj.reshape(-1,3)
        edge_feat_obj_obj=self.e_obj_obj_w(edge_feat_obj_obj)
        edge_feat_obj_obj=edge_feat_obj_obj.reshape(self.n_other_objects,self.n_other_objects)
        adj_ent2ent[self.n_masked_objects:,self.n_masked_objects:]=edge_feat_obj_obj

        edge_feat_obj_msc=edge_feat[self.n_masked_objects:,:self.n_masked_objects]
        edge_feat_obj_msc=edge_feat_obj_msc.reshape(-1,3)
        edge_feat_obj_msc=self.e_obj_msc_w(edge_feat_obj_msc)
        edge_feat_obj_msc=edge_feat_obj_msc.reshape(self.n_other_objects,self.n_masked_objects)
        adj_ent2ent[self.n_masked_objects:,:self.n_masked_objects]=edge_feat_obj_msc
        adj_ent2ent[:self.n_masked_objects,self.n_masked_objects:]=edge_feat_obj_msc.t()

        
        for i in range(self.n_masked_objects):
            msc1_idx=i
            for j in range(i+1, self.n_masked_objects):
                msc2_idx=j
                distance = 1.0 / (node_dist[msc1_idx, msc2_idx] / diag_len + 1.0)
                overlap = node_iou[msc1_idx, msc2_idx]
                visual = node_vis[msc1_idx, msc2_idx]
                e_value = self.e_msc_msc_w(torch.tensor([visual, distance, overlap], device=self.device))
                adj_ent2ent[msc1_idx][msc2_idx]=e_value
                adj_ent2ent[msc2_idx][msc1_idx]=e_value
        
        '''for i in range(self.n_other_objects):
            node_idx = self.n_masked_objects + i
            
            for j in range(self.n_masked_objects):
                msc_idx = j

                distance = 1.0 / (node_dist[node_idx, msc_idx] / diag_len + 1.0)
                overlap = node_iou[node_idx, msc_idx]
                visual = node_vis[node_idx, msc_idx]

                e_value = self.e_obj_msc_w(torch.tensor([visual, distance, overlap], device=self.device))
                
                adj_ent2ent[msc_idx][node_idx]=e_value
                adj_ent2ent[node_idx][msc_idx]=e_value
                
            # object_node : object_node
            for k in range(i+1, self.n_other_objects):
                obj_idx = self.n_masked_objects + k
                
                distance = 1.0 / (node_dist[node_idx, obj_idx] / diag_len + 1.0)
                overlap = node_iou[node_idx, obj_idx]
                visual = node_vis[node_idx, obj_idx]
                e_value = self.e_obj_obj_w(torch.tensor([visual, distance, overlap], device=self.device))

                adj_ent2ent[obj_idx][node_idx]=e_value
                adj_ent2ent[node_idx][obj_idx]=e_value'''
                       
        for i in range(self.n_nodes-1):
            adj_ent2ent[i][i]=1

        return adj_ent2ent, annotation

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
