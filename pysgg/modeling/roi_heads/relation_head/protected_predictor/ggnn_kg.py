##################################################################
# From my_ggnn_15: normalizing messages
##################################################################

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from torch.nn import init

class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, dim_in_hid_out, act_fn='ReLU', last_act=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_in_hid_out) - 1):
            layers.append(XavierLinear(dim_in_hid_out[i], dim_in_hid_out[i + 1]))
            if i < len(dim_in_hid_out) - 2 or last_act:
                layers.append(getattr(torch.nn, act_fn)())
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def wrap(nparr):
    return Variable(torch.from_numpy(nparr).float().cuda(), requires_grad=False)

def arange(num):
    return torch.arange(num).type(torch.LongTensor).cuda()

def normalize(tensor, dim, eps=1e-4):
    return tensor / torch.sqrt(torch.max((tensor**2).sum(dim=dim, keepdim=True), wrap(np.asarray([eps]))))

class GGNN(nn.Module):
    def __init__(self, cfg, emb_path, graph_path, time_step_num=3, hidden_dim=512, output_dim=512, 
                 use_embedding=False, use_knowledge=True, refine_obj_cls=True, top_k_to_keep=5, normalize_messages=True):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num

        self.cfg=cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_event_cls = cfg.EVENT.NUM_CLASSES
        self.num_scene_cls = 365
                
        if use_embedding:
            with open(emb_path, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle.load(fin)
        else:
            self.emb_obj = np.eye(self.num_obj_cls, dtype=np.float32)
            self.emb_eve = np.eye(self.num_event_cls, dtype=np.float32)
            self.emb_sce = np.eye(self.num_scene_cls, dtype=np.float32)

        if use_knowledge:
            kg_path='/workspace/huangyunyi/data/WIDER_v0.1/KG/'
            self.adjmtx_obj2obj = np.load(os.path.join(kg_path,'o1_of_o2.npy'))
            self.adjmtx_obj2obj=np.expand_dims(self.adjmtx_obj2obj, axis=0)
            self.adjmtx_obj2obj = np.pad(self.adjmtx_obj2obj, ((0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0)
            self.adjmtx_obj2eve = np.load(os.path.join(kg_path,'object_of_event.npy'))
            self.adjmtx_obj2eve=np.expand_dims(self.adjmtx_obj2eve, axis=0)
            self.adjmtx_obj2eve = np.pad(self.adjmtx_obj2eve, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
            self.adjmtx_eve2obj = np.load(os.path.join(kg_path,'event_of_objects.npy'))
            self.adjmtx_eve2obj=np.expand_dims(self.adjmtx_eve2obj, axis=0)
            self.adjmtx_eve2obj = np.pad(self.adjmtx_eve2obj, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
            self.adjmtx_eve2obj=np.transpose(self.adjmtx_eve2obj, (0, 2, 1))
            self.adjmtx_eve2eve = np.zeros((1, self.num_event_cls, self.num_event_cls), dtype=np.float32)

        else:
            self.adjmtx_obj2obj = np.zeros((1, self.num_obj_cls, self.num_obj_cls), dtype=np.float32)
            self.adjmtx_obj2eve = np.zeros((1, self.num_obj_cls, self.num_event_cls), dtype=np.float32)
            self.adjmtx_eve2obj = np.zeros((1, self.num_event_cls, self.num_obj_cls), dtype=np.float32)
            self.adjmtx_eve2eve = np.zeros((1, self.num_event_cls, self.num_event_cls), dtype=np.float32) #

            #self.adjmtx_obj2sce = np.zeros((1, self.num_obj_cls, self.num_scene_cls), dtype=np.float32)
            #self.adjmtx_sce2obj = np.zeros((1, self.num_scene_cls, self.num_obj_cls), dtype=np.float32)
            #self.adjmtx_eve2sce = np.zeros((1, self.num_event_cls, self.num_scene_cls), dtype=np.float32)
            #self.adjmtx_sce2eve = np.zeros((1, self.num_scene_cls, self.num_event_cls), dtype=np.float32)

        
        self.num_edge_types_obj2obj = self.adjmtx_obj2obj.shape[0]
        self.num_edge_types_obj2eve = self.adjmtx_obj2eve.shape[0]
        self.num_edge_types_eve2obj = self.adjmtx_eve2obj.shape[0]
        self.num_edge_types_eve2eve = self.adjmtx_eve2obj.shape[0]

        #self.num_edge_types_obj2sce = self.adjmtx_obj2sce.shape[0]
        #self.num_edge_types_sce2obj = self.adjmtx_sce2obj.shape[0]
        #self.num_edge_types_eve2sce = self.adjmtx_eve2sce.shape[0]
        #self.num_edge_types_sce2eve = self.adjmtx_sce2eve.shape[0]
        
        self.fc_init_ont_obj = nn.Linear(self.emb_obj.shape[1], hidden_dim)
        self.fc_init_ont_eve = nn.Linear(self.emb_eve.shape[1], hidden_dim)
        #self.fc_init_ont_sce = nn.Linear(self.emb_sce.shape[1], hidden_dim)
        self.fc_init_img_obj = nn.Linear(4096, hidden_dim)
        self.fc_init_img_eve = nn.Linear(2048, hidden_dim)
        
        self.fc_mp_send_ont_obj = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_ont_eve = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        #self.fc_mp_send_ont_sce = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_obj = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_eve = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        #self.fc_mp_send_img_sce = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        
        self.fc_mp_receive_ont_obj = MLP([(self.num_edge_types_obj2obj + self.num_edge_types_eve2obj + 1) * hidden_dim // 4, 
                                          (self.num_edge_types_obj2obj + self.num_edge_types_eve2obj + 1) * hidden_dim // 4, 
                                          hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_ont_eve = MLP([(self.num_edge_types_obj2eve + self.num_edge_types_eve2eve + 1) * hidden_dim // 4, 
                                           (self.num_edge_types_obj2eve + self.num_edge_types_eve2eve + 1) * hidden_dim // 4, 
                                           hidden_dim], act_fn='ReLU', last_act=True)
        #self.fc_mp_receive_ont_sce = MLP([(self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4, 
        #                                   (self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4, 
        #                                   hidden_dim], act_fn='ReLU', last_act=True)
        
        self.fc_mp_receive_img_obj = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_eve = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        
        self.fc_eq3_w_ont_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_obj = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_ont_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_eve = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_obj = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_obj = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_eve = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_eve = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output_proj_img_eve = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        self.fc_output_proj_ont_eve = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        
        self.refine_obj_cls = refine_obj_cls
        if self.refine_obj_cls:
            self.fc_output_proj_img_obj = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_obj = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)            
        
        self.top_k_to_keep = top_k_to_keep
        self.normalize_messages = normalize_messages
        self.debug_info = {}
        
    def forward(self, obj_logits, roi_feature, scene_info):

        #初始化节点数（ont都是满的）
        num_img_obj = obj_logits.size(0)
        num_ont_obj = self.emb_obj.shape[0]
        num_ont_eve = self.emb_eve.shape[0]
        num_img_eve=1
        
        #初始化节点特征
        nodes_ont_obj = self.fc_init_ont_obj(wrap(self.emb_obj))
        nodes_ont_eve = self.fc_init_ont_eve(wrap(self.emb_eve))        
        nodes_img_obj = self.fc_init_img_obj(roi_feature)
        nodes_img_eve = self.fc_init_img_eve(scene_info['feat'].unsqueeze(0))
        
        #初始化边（邻接矩阵）
        edges_ont_obj2obj = wrap(self.adjmtx_obj2obj)
        edges_ont_obj2eve = wrap(self.adjmtx_obj2eve)
        edges_ont_eve2obj = wrap(self.adjmtx_eve2obj)
        edges_ont_eve2eve = wrap(self.adjmtx_eve2eve)

        #归一化邻接矩阵
        edges_ont_obj2obj = edges_ont_obj2obj / torch.max(edges_ont_obj2obj.sum(dim=1, keepdim=True), wrap(np.asarray([1.0])))
        edges_ont_obj2eve = edges_ont_obj2eve / torch.max(edges_ont_obj2eve.sum(dim=1, keepdim=True), wrap(np.asarray([1.0])))
        edges_ont_eve2obj = edges_ont_eve2obj / torch.max(edges_ont_eve2obj.sum(dim=1, keepdim=True), wrap(np.asarray([1.0])))
        edges_ont_eve2eve = edges_ont_eve2eve / torch.max(edges_ont_eve2eve.sum(dim=1, keepdim=True), wrap(np.asarray([1.0])))
        
        #初始化图中事件与物体的邻接矩阵（全连接）
        edges_img_eve2obj = wrap(np.ones((num_img_eve, num_img_obj)))
        edges_img_obj2eve = edges_img_eve2obj.t()

        edges_img_eve2obj = edges_img_eve2obj / torch.max(edges_img_eve2obj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))

        #图像到KG的边（事件）（是否改成全1？）
        edges_img2ont_eve =  wrap(np.zeros((num_img_eve, num_ont_eve)))
        edges_ont2img_eve = edges_img2ont_eve.t()
        activation_img_eve = wrap(np.zeros((num_img_eve,)))
        
        for t in range(self.time_step_num):
            #根据预测score初始化obj的img2ont边，只保留topK，并归一化
            obj_fg_cls_probs = F.softmax(obj_logits[:, 1:], dim=1)
            edges_img2ont_obj = torch.cat([wrap(np.zeros([obj_fg_cls_probs.size(0), 1])), obj_fg_cls_probs], dim=1)
            edges_img2ont_obj.scatter_(1, torch.topk(edges_img2ont_obj, num_ont_obj - self.top_k_to_keep, dim=1, largest=False, sorted=False)[1], 0.0)
            edges_ont2img_obj = edges_img2ont_obj.t()
            edges_img2ont_obj = edges_img2ont_obj / torch.max(edges_img2ont_obj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
            
            

            #消息传播
            message_send_ont_obj = self.fc_mp_send_ont_obj(nodes_ont_obj)
            message_send_ont_eve = self.fc_mp_send_ont_eve(nodes_ont_eve)
            message_send_img_obj = self.fc_mp_send_img_obj(nodes_img_obj)
            message_send_img_eve = self.fc_mp_send_img_eve(nodes_img_eve)

            message_incoming_ont_obj = torch.stack(
                [torch.mm(edges_ont_obj2obj[i].t(), message_send_ont_obj) for i in range(self.num_edge_types_obj2obj)] +
                [torch.mm(edges_ont_eve2obj[i].t(), message_send_ont_eve) for i in range(self.num_edge_types_eve2obj)] +
                [torch.mm(edges_img2ont_obj.t(), message_send_img_obj),]
            , 1)
            
            message_incoming_ont_eve = torch.stack(
                [torch.mm(edges_ont_obj2eve[i].t(), message_send_ont_obj) for i in range(self.num_edge_types_obj2eve)] +
                [torch.mm(edges_ont_eve2eve[i].t(), message_send_ont_eve) for i in range(self.num_edge_types_eve2eve)] +
                [torch.mm(edges_img2ont_eve.t(), message_send_img_eve),]
            , 1)
            
            message_incoming_img_obj = torch.stack([
                torch.mm(edges_img_eve2obj.t(), message_send_img_eve),
                torch.mm(edges_img_eve2obj.t(), message_send_img_eve),
                torch.mm(edges_ont2img_obj.t(), message_send_ont_obj),
            ], 1)
            
            message_incoming_img_eve = torch.stack([
                torch.mm(edges_img_obj2eve.t(), message_send_img_obj),
                torch.mm(edges_img_obj2eve.t(), message_send_img_obj),
                torch.mm(edges_ont2img_eve.t(), message_send_ont_eve),
            ], 1)
            '''
            self.debug_info[f'incoming_message_size_{t}'] = [
                torch.pow(message_incoming_ont_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_ont_pred, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_pred, 2).sum(2).mean(0),
            ]
            '''
            
            if self.normalize_messages:
                message_incoming_ont_obj = normalize(message_incoming_ont_obj, 2)
                message_incoming_ont_eve = normalize(message_incoming_ont_eve, 2)
                message_incoming_img_obj = normalize(message_incoming_img_obj, 2)
                message_incoming_img_eve = normalize(message_incoming_img_eve, 2)
            
            '''
            self.debug_info[f'incoming_message_size_post_normalization_{t}'] = [
                torch.pow(message_incoming_ont_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_ont_pred, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_pred, 2).sum(2).mean(0),
            ]
            '''
            message_received_ont_obj = self.fc_mp_receive_ont_obj(message_incoming_ont_obj.view(num_ont_obj, -1))            
            message_received_ont_eve = self.fc_mp_receive_ont_eve(message_incoming_ont_eve.view(num_ont_eve, -1))            
            message_received_img_obj = self.fc_mp_receive_img_obj(message_incoming_img_obj.view(num_img_obj, -1))            
            message_received_img_eve = self.fc_mp_receive_img_eve(message_incoming_img_eve.view(num_img_eve, -1))

            '''
            self.debug_info[f'received_message_size_{t}'] = [
                torch.pow(message_received_ont_ent, 2).sum(1).mean(0),
                torch.pow(message_received_ont_pred, 2).sum(1).mean(0),
                torch.pow(message_received_img_ent, 2).sum(1).mean(0),
                torch.pow(message_received_img_pred, 2).sum(1).mean(0),
            ]
            '''
            
            z_ont_obj = torch.sigmoid(self.fc_eq3_w_ont_obj(message_received_ont_obj) + self.fc_eq3_u_ont_obj(nodes_ont_obj))
            r_ont_obj = torch.sigmoid(self.fc_eq4_w_ont_obj(message_received_ont_obj) + self.fc_eq4_u_ont_obj(nodes_ont_obj))
            h_ont_obj = torch.tanh(self.fc_eq5_w_ont_obj(message_received_ont_obj) + self.fc_eq5_u_ont_obj(r_ont_obj * nodes_ont_obj))
            nodes_ont_obj_new = (1 - z_ont_obj) * nodes_ont_obj + z_ont_obj * h_ont_obj

            z_ont_eve = torch.sigmoid(self.fc_eq3_w_ont_eve(message_received_ont_eve) + self.fc_eq3_u_ont_eve(nodes_ont_eve))
            r_ont_eve = torch.sigmoid(self.fc_eq4_w_ont_eve(message_received_ont_eve) + self.fc_eq4_u_ont_eve(nodes_ont_eve))
            h_ont_eve = torch.tanh(self.fc_eq5_w_ont_eve(message_received_ont_eve) + self.fc_eq5_u_ont_eve(r_ont_eve * nodes_ont_eve))
            nodes_ont_eve_new = (1 - z_ont_eve) * nodes_ont_eve + z_ont_eve * h_ont_eve

            z_img_obj = torch.sigmoid(self.fc_eq3_w_img_obj(message_received_img_obj) + self.fc_eq3_u_img_obj(nodes_img_obj))
            r_img_obj = torch.sigmoid(self.fc_eq4_w_img_obj(message_received_img_obj) + self.fc_eq4_u_img_obj(nodes_img_obj))
            h_img_obj = torch.tanh(self.fc_eq5_w_img_obj(message_received_img_obj) + self.fc_eq5_u_img_obj(r_img_obj * nodes_img_obj))
            nodes_img_obj_new = (1 - z_img_obj) * nodes_img_obj + z_img_obj * h_img_obj

            z_img_eve = torch.sigmoid(self.fc_eq3_w_img_eve(message_received_img_eve) + self.fc_eq3_u_img_eve(nodes_img_eve))
            r_img_eve = torch.sigmoid(self.fc_eq4_w_img_eve(message_received_img_eve) + self.fc_eq4_u_img_eve(nodes_img_eve))
            h_img_eve = torch.tanh(self.fc_eq5_w_img_eve(message_received_img_eve) + self.fc_eq5_u_img_eve(r_img_eve * nodes_img_eve))
            nodes_img_eve_new = (1 - z_img_eve) * nodes_img_eve + z_img_eve * h_img_eve

            relative_state_change_ont_obj = torch.sum(torch.abs(nodes_ont_obj_new - nodes_ont_obj)) / torch.sum(torch.abs(nodes_ont_obj))
            relative_state_change_ont_eve = torch.sum(torch.abs(nodes_ont_eve_new - nodes_ont_eve)) / torch.sum(torch.abs(nodes_ont_eve))
            relative_state_change_img_obj = torch.sum(torch.abs(nodes_img_obj_new - nodes_img_obj)) / torch.sum(torch.abs(nodes_img_obj))
            relative_state_change_img_eve = torch.sum(torch.abs(nodes_img_eve_new - nodes_img_eve)) / torch.sum(torch.abs(nodes_img_eve))
        
            self.debug_info[f'relative_state_change_{t}'] = [
                relative_state_change_ont_obj, 
                relative_state_change_ont_eve, 
                relative_state_change_img_obj, 
                relative_state_change_img_eve
            ]
        
            nodes_ont_obj = nodes_ont_obj_new
            nodes_ont_eve = nodes_ont_eve_new
            nodes_img_obj = nodes_img_obj_new
            nodes_img_eve = nodes_img_eve_new
            
            eve_cls_logits = torch.mm(self.fc_output_proj_img_eve(nodes_img_eve), self.fc_output_proj_ont_eve(nodes_ont_eve).t())

            eve_fg_cls_probs = F.softmax(eve_cls_logits[:, 1:], dim=1)
            edges_img2ont_eve = torch.cat([wrap(np.zeros([eve_fg_cls_probs.size(0), 1])), eve_fg_cls_probs], dim=1)
                        
            edges_img2ont_eve.scatter_(1, torch.topk(edges_img2ont_eve, num_ont_eve - self.top_k_to_keep, dim=1, largest=False, sorted=False)[1], 0.0)
            edges_ont2img_eve = edges_img2ont_eve.t()            
            edges_img2ont_eve = edges_img2ont_eve / torch.max(edges_img2ont_eve.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
            
            if self.refine_obj_cls:
                obj_cls_logits = torch.mm(self.fc_output_proj_img_obj(nodes_img_obj), self.fc_output_proj_ont_obj(nodes_ont_obj).t())
                
        return eve_cls_logits, obj_cls_logits

