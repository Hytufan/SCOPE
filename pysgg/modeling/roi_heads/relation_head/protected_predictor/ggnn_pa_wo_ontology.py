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
import pandas as pd
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
#import clip
from CLIP import clip
import torchvision.transforms as transforms
from .privacy_score_v2 import PrivacyScore

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class QKVAttention(nn.Module):
    def __init__(self, node_feature_dim=2048, event_feature_dim=2048, attention_hidden_dim=2048):
        super(QKVAttention, self).__init__()
        # 定义 Q, K, V 投影层
        self.query_proj = nn.Linear(event_feature_dim, attention_hidden_dim)
        self.key_proj = nn.Linear(node_feature_dim, attention_hidden_dim)
        self.value_proj = nn.Linear(node_feature_dim, attention_hidden_dim)
    
    def forward(self, nodes_obj, nodes_eve):
        # Q, K, V 投影
        query = self.query_proj(nodes_eve)  # (1, attention_hidden_dim)
        key = self.key_proj(nodes_obj)      # (num_nodes, attention_hidden_dim)
        value = self.value_proj(nodes_obj)  # (num_nodes, attention_hidden_dim)

        # 计算注意力分数 (QK 点积)
        attention_scores = torch.matmul(query, key.T)  # (1, num_nodes)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (1, num_nodes)

        # 加权求和得到全局特征
        global_feature = torch.matmul(attention_weights, value)  # (1, attention_hidden_dim)

        return global_feature, attention_weights


class GGNN(nn.Module):
    def __init__(self, cfg, time_step_num=3, hidden_dim=512, output_dim=512, emb_path=None, graph_path=None,
                 use_knowledge=False, refine_obj_cls=True, top_k_to_keep=5, normalize_messages=True):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num

        self.cfg=cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_event_cls = cfg.EVENT.NUM_CLASSES
        self.num_scene_cls = 365
        self.obj_th = cfg.KGEVENT.OBJ_TH
        self.class_map=cfg.PROTECTED.PROTECT_CLASSES
        self.use_clip=cfg.KGEVENT.CLIP
        self.learnable_adj=cfg.KGEVENT.LEARNABLE_ADJ
        if len(self.class_map)==0:
            self.class_map=list(range(151))
        self.obj_and_event=True
        self.privacy_score=cfg.KGEVENT.PRIVACY_SCORE
        self.obj_clip=False
        self.multi_label=cfg.PROTECTED.MULTI_LABEL
        self.use_event_model=False
        if self.use_event_model:
            self.event_model = resnet50(pretrained=True)
            self.event_model.fc = torch.nn.Linear(self.event_model.fc.in_features, self.num_event_cls)
            self.event_model.to(torch.cuda.current_device())
            best_model_weights = torch.load('/workspace/huangyunyi/code/resnet50/resnet50/ckpt/mydata_ori_0.6510263929618768.pth',map_location=f'cuda:{torch.cuda.current_device()}')
            self.event_model.load_state_dict(best_model_weights)
            self.event_model = torch.nn.Sequential(*list(self.event_model.children())[:-1])
            self.event_model.eval()

        if self.use_clip:
            eve_file_path = cfg.KGEVENT.EVENT_CLASSES
            eve_label_list = []
            with open(eve_file_path, 'r') as file:
                for line in file:
                    # 按顺序读取每行，并将下划线替换为空格
                    number, string = line.strip().split(" ", 1)
                    modified_string = string.replace("_", " ")
                    eve_label_list.append(modified_string)

            obj_file_path = "/workspace/huangyunyi/res/obj_class.txt"
            obj_label_list = []
            with open(obj_file_path, 'r') as file:
                for line in file:
                    # 按顺序读取每行，并将下划线替换为空格
                    number, string = line.strip().split(" ", 1)
                    modified_string = string.replace("_", " ")
                    obj_label_list.append(modified_string)

            sce_file_path = "/workspace/huangyunyi/res/sce_class.txt"
            sce_label_list = []
            with open(sce_file_path, 'r') as file:
                for line in file:
                    # 按顺序读取每行，并将下划线替换为空格
                    string, number = line.strip().split(" ", 1)
                    modified_string = string.replace("_", " ").replace('/',' ')[3:]
                    sce_label_list.append(modified_string)
            #print(eve_label_list,obj_label_list,sce_label_list)

                    
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32',device=torch.cuda.current_device())
            eve_text_inputs = torch.cat([clip.tokenize(f"{c}") for c in eve_label_list]).to(torch.cuda.current_device())
            obj_text_inputs = torch.cat([clip.tokenize(f"{c}") for c in obj_label_list]).to(torch.cuda.current_device())
            sce_text_inputs = torch.cat([clip.tokenize(f"{c}") for c in sce_label_list]).to(torch.cuda.current_device())

            #print(self.clip_model.device,eve_text_inputs.device)
            with torch.no_grad():
                self.emb_eve = self.clip_model.encode_text(eve_text_inputs)
                self.emb_obj = self.clip_model.encode_text(obj_text_inputs)
                self.emb_sce = self.clip_model.encode_text(sce_text_inputs)

            self.emb_eve=self.emb_eve.cpu().numpy()
            self.emb_obj=self.emb_obj.cpu().numpy()
            self.emb_sce=self.emb_sce.cpu().numpy()

        else:     
            if cfg.PROTECTED.USE_WORD_EMBEDDING:

                df = pd.read_csv('/workspace/huangyunyi/data/WIDER_v0.1/KG/obj.csv')
                embedding_columns = df.columns[1:]
                embedding_data = df[embedding_columns]
                self.emb_obj = embedding_data.to_numpy()
                zero_row = np.zeros((1, df.shape[1]-1))
                self.emb_obj = np.vstack((zero_row, self.emb_obj))

                df = pd.read_csv('/workspace/huangyunyi/data/WIDER_v0.1/KG/scene.csv')
                embedding_columns = df.columns[1:]
                embedding_data = df[embedding_columns]
                self.emb_sce = embedding_data.to_numpy()

                df = pd.read_csv('/workspace/huangyunyi/data/WIDER_v0.1/KG/event.csv')
                embedding_columns = df.columns[1:]
                embedding_data = df[embedding_columns]
                self.emb_eve = embedding_data.to_numpy()
                
            else:
                self.emb_obj = np.eye(self.num_obj_cls, dtype=np.float32)
                self.emb_eve = np.eye(self.num_event_cls, dtype=np.float32)
                self.emb_sce = np.eye(self.num_scene_cls, dtype=np.float32)

        if use_knowledge:
            kg_path=cfg.KGEVENT.KG_PATH #'/workspace/huangyunyi/data/WIDER_v0.1/KG/vise_wider_npy/'
            if kg_path=='/workspace/huangyunyi/data/WIDER_v0.1/KG/vise_wider_npy/':
                self.adjmtx_obj2obj = np.load(os.path.join(kg_path,'o1_of_o2.npy'))
                self.adjmtx_obj2obj = np.expand_dims(self.adjmtx_obj2obj, axis=0)
                self.adjmtx_obj2obj = np.pad(self.adjmtx_obj2obj, ((0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_obj2eve = np.load(os.path.join(kg_path,'object_of_event.npy'))
                self.adjmtx_obj2eve = np.expand_dims(self.adjmtx_obj2eve, axis=0)
                self.adjmtx_obj2eve = np.pad(self.adjmtx_obj2eve, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_eve2obj = np.load(os.path.join(kg_path,'event_of_objects.npy'))
                self.adjmtx_eve2obj = np.expand_dims(self.adjmtx_eve2obj, axis=0)
                self.adjmtx_eve2obj = np.pad(self.adjmtx_eve2obj, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                self.adjmtx_eve2obj = np.transpose(self.adjmtx_eve2obj, (0, 2, 1))

                self.adjmtx_obj2sce = np.load(os.path.join(kg_path,'object_of_scenes.npy'))
                self.adjmtx_obj2sce = np.expand_dims(self.adjmtx_obj2sce, axis=0)
                self.adjmtx_obj2sce = np.pad(self.adjmtx_obj2sce, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_sce2obj = np.load(os.path.join(kg_path,'scene_of_objects.npy'))
                self.adjmtx_sce2obj = np.expand_dims(self.adjmtx_sce2obj, axis=0)
                self.adjmtx_sce2obj = np.pad(self.adjmtx_sce2obj, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                self.adjmtx_sce2obj = np.transpose(self.adjmtx_sce2obj, (0, 2, 1))
                
                #self.adjmtx_eve2eve = np.zeros((1, self.num_event_cls, self.num_event_cls), dtype=np.float32)

                self.adjmtx_sce2eve = np.load(os.path.join(kg_path,'scene_of_event.npy'))
                self.adjmtx_sce2eve = np.expand_dims(self.adjmtx_sce2eve, axis=0)
                self.adjmtx_sce2eve = np.transpose(self.adjmtx_sce2eve, (0, 2, 1))
                self.adjmtx_eve2sce = np.load(os.path.join(kg_path,'event_of_scene.npy'))
                self.adjmtx_eve2sce = np.expand_dims(self.adjmtx_eve2sce, axis=0)

            else:
                self.adjmtx_obj2obj = np.load(os.path.join(kg_path,'o1_of_o2.npy'))
                self.adjmtx_obj2obj = np.expand_dims(self.adjmtx_obj2obj, axis=0)
                #self.adjmtx_obj2obj = np.pad(self.adjmtx_obj2obj, ((0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_obj2eve = np.load(os.path.join(kg_path,'object_of_event.npy'))
                self.adjmtx_obj2eve = np.expand_dims(self.adjmtx_obj2eve, axis=0)
                #self.adjmtx_obj2eve = np.pad(self.adjmtx_obj2eve, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_eve2obj = np.load(os.path.join(kg_path,'event_of_objects.npy'))
                self.adjmtx_eve2obj = np.expand_dims(self.adjmtx_eve2obj, axis=0)
                #self.adjmtx_eve2obj = np.pad(self.adjmtx_eve2obj, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                #self.adjmtx_eve2obj = np.transpose(self.adjmtx_eve2obj, (0, 2, 1))

                self.adjmtx_obj2sce = np.load(os.path.join(kg_path,'object_of_scenes.npy'))
                self.adjmtx_obj2sce = np.expand_dims(self.adjmtx_obj2sce, axis=0)
                #self.adjmtx_obj2sce = np.pad(self.adjmtx_obj2sce, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                
                self.adjmtx_sce2obj = np.load(os.path.join(kg_path,'scene_of_objects.npy'))
                self.adjmtx_sce2obj = np.expand_dims(self.adjmtx_sce2obj, axis=0)
                #self.adjmtx_sce2obj = np.pad(self.adjmtx_sce2obj, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
                #self.adjmtx_sce2obj = np.transpose(self.adjmtx_sce2obj, (0, 2, 1))
                
                #self.adjmtx_eve2eve = np.zeros((1, self.num_event_cls, self.num_event_cls), dtype=np.float32)

                self.adjmtx_sce2eve = np.load(os.path.join(kg_path,'scene_of_event.npy'))
                self.adjmtx_sce2eve = np.expand_dims(self.adjmtx_sce2eve, axis=0)
                #self.adjmtx_sce2eve = np.transpose(self.adjmtx_sce2eve, (0, 2, 1))
                self.adjmtx_eve2sce = np.load(os.path.join(kg_path,'event_of_scene.npy'))
                self.adjmtx_eve2sce = np.expand_dims(self.adjmtx_eve2sce, axis=0)

        else:
            self.adjmtx_obj2obj = np.zeros((1, self.num_obj_cls, self.num_obj_cls), dtype=np.float32)
            self.adjmtx_obj2eve = np.zeros((1, self.num_obj_cls, self.num_event_cls), dtype=np.float32)
            self.adjmtx_eve2obj = np.zeros((1, self.num_event_cls, self.num_obj_cls), dtype=np.float32)
            #self.adjmtx_eve2eve = np.zeros((1, self.num_event_cls, self.num_event_cls), dtype=np.float32) #

            self.adjmtx_obj2sce = np.zeros((1, self.num_obj_cls, self.num_scene_cls), dtype=np.float32)
            self.adjmtx_sce2obj = np.zeros((1, self.num_scene_cls, self.num_obj_cls), dtype=np.float32)
            self.adjmtx_eve2sce = np.zeros((1, self.num_event_cls, self.num_scene_cls), dtype=np.float32)
            self.adjmtx_sce2eve = np.zeros((1, self.num_scene_cls, self.num_event_cls), dtype=np.float32)


        # 将本体图的边作为模型参数
        if self.learnable_adj:
            self.adjmtx_obj2obj=torch.nn.Parameter(torch.tensor(self.adjmtx_obj2obj, dtype=torch.float32), requires_grad=True)
            self.adjmtx_obj2sce=torch.nn.Parameter(torch.tensor(self.adjmtx_obj2sce, dtype=torch.float32), requires_grad=True)
            self.adjmtx_obj2eve=torch.nn.Parameter(torch.tensor(self.adjmtx_obj2eve, dtype=torch.float32), requires_grad=True)
            self.adjmtx_sce2obj=torch.nn.Parameter(torch.tensor(self.adjmtx_sce2obj, dtype=torch.float32), requires_grad=True)
            self.adjmtx_eve2obj=torch.nn.Parameter(torch.tensor(self.adjmtx_eve2obj, dtype=torch.float32), requires_grad=True)
            self.adjmtx_sce2eve=torch.nn.Parameter(torch.tensor(self.adjmtx_sce2eve, dtype=torch.float32), requires_grad=True)
            self.adjmtx_eve2sce=torch.nn.Parameter(torch.tensor(self.adjmtx_eve2sce, dtype=torch.float32), requires_grad=True)
            
        self.num_edge_types_obj2obj = self.adjmtx_obj2obj.shape[0]
        self.num_edge_types_obj2eve = self.adjmtx_obj2eve.shape[0]
        self.num_edge_types_eve2obj = self.adjmtx_eve2obj.shape[0]
        #self.num_edge_types_eve2eve = self.adjmtx_eve2obj.shape[0]
        self.num_edge_types_sce2obj = self.adjmtx_sce2obj.shape[0]
        self.num_edge_types_obj2sce = self.adjmtx_obj2sce.shape[0]

        #self.num_edge_types_obj2sce = self.adjmtx_obj2sce.shape[0]
        #self.num_edge_types_sce2obj = self.adjmtx_sce2obj.shape[0]
        self.num_edge_types_eve2sce = self.adjmtx_eve2sce.shape[0]
        self.num_edge_types_sce2eve = self.adjmtx_sce2eve.shape[0]
        
        self.fc_init_img_obj = nn.Linear(512 if self.obj_clip else 4096, hidden_dim)
        self.fc_init_img_eve = nn.Linear(512 if self.use_clip else 2048, hidden_dim)
        self.fc_init_img_sce = nn.Linear(512 if self.obj_clip else 4096, hidden_dim)
        self.fc_init_img_msk = nn.Linear(512 if self.obj_clip else 4096, hidden_dim)
        
        self.fc_mp_send_img_obj = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_eve = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_sce = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_msk = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        
        self.fc_mp_receive_img_obj = MLP([4 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_eve = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_sce = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_msk = MLP([4 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)

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

        self.fc_eq3_w_img_sce = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_sce = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_sce = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_sce = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_sce = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_sce = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_msk = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_msk = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_msk = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_msk = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_msk = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_msk = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output_proj_img_eve = MLP([hidden_dim, hidden_dim, self.num_event_cls], act_fn='ReLU', last_act=False)
        
        self.refine_obj_cls = refine_obj_cls
        #if self.refine_obj_cls:
        self.fc_output_proj_img_msk = MLP([hidden_dim, hidden_dim, len(self.class_map)], act_fn='ReLU', last_act=False)

        #self.fc_output_img_msk = MLP([hidden_dim, hidden_dim // 4, len(self.class_map)], act_fn='ReLU', last_act=False)       
        
        self.top_k_to_keep = top_k_to_keep
        self.normalize_messages = normalize_messages
        self.debug_info = {}  

        self.PrivacyScore=PrivacyScore(cfg=cfg)
        
        self.use_attention=''


        
    def forward(self, nodes_feat, proposal, protect_target, scene_info, adj_matrix, image=None, mask_adj=None):

        device=nodes_feat.device
        obj_logits=proposal.get_field("predict_logits")
        #print(f'4 {type(image)},{image.shape}')

        '''pred_scores=proposal.get_field('pred_scores')
        indices = torch.where(pred_scores > self.obj_th)
        if pred_scores[indices].size(0)==0:
            indices = torch.where(pred_scores >= torch.max(pred_scores)-0.001)
        _, indices = torch.topk(pred_scores, k=min(20,pred_scores.size(0)))
        obj_logits=obj_logits[indices]
        roi_feature=roi_feature[indices]'''

        #print(obj_logits)
        #初始化节点数（ont都是满的）
        num_img_obj = obj_logits.size(0)
        num_img_msk = nodes_feat.shape[0]-num_img_obj
        num_img_eve = 1
        num_img_sce = 1

        if self.use_clip:
            to_pil = transforms.ToPILImage()
            eve_image_input = self.clip_preprocess(to_pil(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                eve_image_features = self.clip_model.encode_image(eve_image_input)
            nodes_img_eve = self.fc_init_img_eve(eve_image_features)
        elif self.use_event_model:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            eve_image_input = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                eve_image_features = self.event_model(eve_image_input).squeeze().unsqueeze(0)
            nodes_img_eve = self.fc_init_img_eve(eve_image_features)
            #nodes_img_eve=nodes_img_eve_init.clone()
        else:
            nodes_img_eve = self.fc_init_img_eve(scene_info['feat'].unsqueeze(0))
            
        if mask_adj==None:
            mask_adj=wrap(np.ones(num_img_obj)) 
         
        if self.obj_clip:
            to_pil = transforms.ToPILImage()
            pil_image=to_pil(image)
            proposal_resize=proposal.resize(pil_image.size)
            cropped_images = []
            bboxes = proposal_resize.bbox.cpu().numpy()
            protect_bbox=protect_target.resize(pil_image.size).bbox.cpu().numpy()
            bboxes=np.concatenate([bboxes,protect_bbox],axis=0)
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                x_min, y_min, x_max, y_max = max(0, int(x_min)), max(0, int(y_min)), min(pil_image.width, int(x_max)), min(pil_image.height, int(y_max))
                
                if x_min >= x_max:
                    if x_max > 1:
                        x_min = x_max - 1
                    else:
                        x_max = x_min + 1

                if y_min >= y_max:
                    if y_max > 1:
                        y_min = y_max - 1
                    else:
                        y_max = y_min + 1
                    
                # 裁剪图像
                if x_min >= x_max or y_min >= y_max or x_max>pil_image.width or y_max>pil_image.height or x_min<0 or y_min<0:
                    print("Invalid bounding box:", bbox)
                    print(protect_target.get_field('file_name'))
                cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
                
                # 将裁剪后的图像添加到列表中
                cropped_images.append(cropped_image)

            # 批量预处理
            cropped_image_inputs = [self.clip_preprocess(img) for img in cropped_images]
            cropped_image_inputs = torch.stack(cropped_image_inputs).to(device)

            # 批量特征提取
            with torch.no_grad():
                obj_image_features = self.clip_model.encode_image(cropped_image_inputs)
            
            nodes_img_msk = self.fc_init_img_msk(obj_image_features[-num_img_msk:,:])
            nodes_img_obj = self.fc_init_img_obj(obj_image_features[:-num_img_msk,:]* mask_adj.view(-1, 1))
            nodes_img_sce = self.fc_init_img_sce(eve_image_features)   
        
        else:
            #nodes_img_sce = self.fc_init_img_sce(scene_info['feat'].unsqueeze(0))
            nodes_img_msk = self.fc_init_img_msk(nodes_feat[:num_img_msk,:])
            nodes_img_obj = self.fc_init_img_obj(nodes_feat[num_img_msk:,:]* mask_adj.view(-1, 1)) #* mask_adj.view(-1, 1)

            nodes_img_sce = torch.zeros([self.cfg.PROTECTED.GGNN_D_ANNOTATION], device=device)
            s_feat_dim=scene_info['feat'].shape[-1]
            nodes_img_sce[:s_feat_dim] = scene_info['feat']
            nodes_img_sce[s_feat_dim:(s_feat_dim + self.emb_sce.shape[-1])] = torch.from_numpy(self.emb_sce[scene_info['labels'][0]])
            nodes_img_sce=self.fc_init_img_sce(nodes_img_sce.unsqueeze(0))
    

        #初始化(图中)邻接矩阵
        edges_img_eve2obj = wrap(np.ones((num_img_eve, num_img_obj)))
        edges_img_eve2sce = wrap(np.ones((num_img_eve, num_img_sce)))
        edges_img_obj2sce = wrap(np.ones((num_img_obj, num_img_sce)))
        edges_img_eve2obj = edges_img_eve2obj / torch.max(edges_img_eve2obj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_eve2sce = edges_img_eve2sce / torch.max(edges_img_eve2sce.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_obj2sce = edges_img_obj2sce / torch.max(edges_img_obj2sce.sum(dim=0, keepdim=True), wrap(np.asarray([1.0]))) 
        edges_img_obj2eve = edges_img_eve2obj.t()
        edges_img_sce2eve = edges_img_eve2sce.t()
        edges_img_sce2obj = edges_img_obj2sce.t()

        # mask 节点边初始化（双向版）
        edges_img_msk2sce = wrap(np.ones((num_img_msk, num_img_sce)))
        edges_img_msk2eve = wrap(np.ones((num_img_msk, num_img_eve)))
        edges_img_msk2sce = edges_img_msk2sce / torch.max(edges_img_msk2sce.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_msk2eve = edges_img_msk2eve / torch.max(edges_img_msk2eve.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_sce2msk = edges_img_msk2sce.t()
        edges_img_eve2msk = edges_img_msk2eve.t()
        
        # 不更新版
        '''edges_img_obj2obj = adj_matrix[num_img_msk:,num_img_msk:] #wrap(np.ones((num_img_obj, num_img_obj)))
        edges_img_msk2msk = adj_matrix[:num_img_msk,:num_img_msk]
        edges_img_msk2obj = adj_matrix[:num_img_msk,num_img_msk:]
        edges_img_obj2msk = adj_matrix[num_img_msk:,:num_img_msk]'''

        # 更新版
        edges_img_obj2obj = adj_matrix[num_img_msk:,num_img_msk:]
        edges_img_msk2msk = adj_matrix[:num_img_msk,:num_img_msk]
        edges_img_msk2obj = adj_matrix[:num_img_msk,num_img_msk:]
        edges_img_obj2msk = adj_matrix[num_img_msk:,:num_img_msk]
        #print(edges_img_obj2obj,edges_img_obj2obj.requires_grad)

        '''# mask 节点边初始化（单向版）
        edges_img_msk2sce = wrap(np.zeros((num_img_msk, num_img_sce)))
        edges_img_msk2eve = wrap(np.zeros((num_img_msk, num_img_eve)))
        edges_img_msk2sce = edges_img_msk2sce / torch.max(edges_img_msk2sce.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_msk2eve = edges_img_msk2eve / torch.max(edges_img_msk2eve.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))

        edges_img_sce2msk = wrap(np.ones((num_img_sce,num_img_msk)))
        edges_img_eve2msk = wrap(np.ones((num_img_eve,num_img_msk)))
        edges_img_sce2msk = edges_img_sce2msk / torch.max(edges_img_sce2msk.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_eve2msk = edges_img_eve2msk / torch.max(edges_img_eve2msk.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        
        edges_img_obj2obj = adj_matrix[num_img_msk:,num_img_msk:] #wrap(np.ones((num_img_obj, num_img_obj)))
        edges_img_msk2msk = wrap(np.zeros((num_img_msk, num_img_msk)))
        edges_img_msk2obj = wrap(np.zeros((num_img_msk, num_img_obj)))
        edges_img_obj2msk = adj_matrix[num_img_msk:,:num_img_msk]'''


        '''# 提前更新边
        eve_cls_logits = torch.mm(self.fc_output_proj_img_eve(nodes_img_eve), self.fc_output_proj_ont_eve(nodes_ont_eve).t())
        #eve_logits_steps.append(eve_cls_logits)
        eve_fg_cls_probs = F.softmax(eve_cls_logits, dim=1)
        edges_img2ont_eve = eve_fg_cls_probs.clone() #torch.cat([wrap(np.zeros([eve_fg_cls_probs.size(0), 1])), eve_fg_cls_probs], dim=1)
        edges_img2ont_eve.scatter_(1, torch.topk(edges_img2ont_eve, num_ont_eve - self.top_k_to_keep, dim=1, largest=False, sorted=False)[1], 0.0)            
        edges_img2ont_eve = edges_img2ont_eve / torch.max(edges_img2ont_eve.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_ont2img_eve = edges_img2ont_eve.t()'''

        #eve_logits_steps=[]

        
        for t in range(self.time_step_num):
            '''#根据预测score初始化obj的img2ont边，只保留topK，并归一化
            obj_fg_cls_probs = F.softmax(obj_logits[:, 1:], dim=1)
            edges_img2ont_obj = torch.cat([wrap(np.zeros([obj_fg_cls_probs.size(0), 1])), obj_fg_cls_probs], dim=1)
            edges_img2ont_obj.scatter_(1, torch.topk(edges_img2ont_obj, num_ont_obj - self.top_k_to_keep, dim=1, largest=False, sorted=False)[1], 0.0)
            edges_ont2img_obj = edges_img2ont_obj.t()
            edges_img2ont_obj = edges_img2ont_obj / torch.max(edges_img2ont_obj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))

            sce_fg_cls_probs = scene_info['scores'].unsqueeze(0).to(device)
            #edges_img2ont_sce = torch.cat([wrap(np.zeros([sce_fg_cls_probs.size(0), 1])), sce_fg_cls_probs], dim=1)
            edges_img2ont_sce = sce_fg_cls_probs.clone()
            edges_img2ont_sce.scatter_(1, torch.topk(edges_img2ont_sce, num_ont_sce - self.top_k_to_keep, dim=1, largest=False, sorted=False)[1], 0.0)
            edges_ont2img_sce = edges_img2ont_sce.t()
            edges_img2ont_sce = edges_img2ont_sce / torch.max(edges_img2ont_sce.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))'''

            message_send_img_obj = self.fc_mp_send_img_obj(nodes_img_obj)
            message_send_img_eve = self.fc_mp_send_img_eve(nodes_img_eve)
            message_send_img_sce = self.fc_mp_send_img_sce(nodes_img_sce)
            message_send_img_msk = self.fc_mp_send_img_msk(nodes_img_msk)

            message_incoming_img_obj = torch.stack([
                torch.mm(edges_img_obj2obj.t(), message_send_img_obj),
                torch.mm(edges_img_eve2obj.t(), message_send_img_eve),
                torch.mm(edges_img_sce2obj.t(), message_send_img_sce),
                torch.mm(edges_img_msk2obj.t(), message_send_img_msk),
            ], 1)
            
            message_incoming_img_eve = torch.stack([
                torch.mm(edges_img_obj2eve.t(), message_send_img_obj),
                torch.mm(edges_img_sce2eve.t(), message_send_img_sce),
                torch.mm(edges_img_msk2eve.t(), message_send_img_msk),
            ], 1)

            message_incoming_img_sce = torch.stack([
                torch.mm(edges_img_obj2sce.t(), message_send_img_obj),
                torch.mm(edges_img_eve2sce.t(), message_send_img_eve),
                torch.mm(edges_img_msk2sce.t(), message_send_img_msk),
            ], 1)

            message_incoming_img_msk = torch.stack([
                torch.mm(edges_img_obj2msk.t(), message_send_img_obj),
                torch.mm(edges_img_eve2msk.t(), message_send_img_eve),
                torch.mm(edges_img_sce2msk.t(), message_send_img_sce),
                torch.mm(edges_img_msk2msk.t(), message_send_img_msk),
                #torch.mm(edges_ont2img_msk.t(), message_send_ont_obj),
            ], 1)

            #print(f'{t} mess_in:{message_incoming_img_msk}')
            #print(f'img_obj:{message_send_img_obj}  edge:{edges_img_obj2msk.t()}')

            '''
            self.debug_info[f'incoming_message_size_{t}'] = [
                torch.pow(message_incoming_ont_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_ont_pred, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_pred, 2).sum(2).mean(0),
            ]
            '''
            
            if self.normalize_messages:
                message_incoming_img_obj = normalize(message_incoming_img_obj, 2)
                message_incoming_img_eve = normalize(message_incoming_img_eve, 2)
                message_incoming_img_sce = normalize(message_incoming_img_sce, 2)
                message_incoming_img_msk = normalize(message_incoming_img_msk, 2)
            
            '''
            self.debug_info[f'incoming_message_size_post_normalization_{t}'] = [
                torch.pow(message_incoming_ont_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_ont_pred, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_ent, 2).sum(2).mean(0),
                torch.pow(message_incoming_img_pred, 2).sum(2).mean(0),
            ]
            '''         
            message_received_img_obj = self.fc_mp_receive_img_obj(message_incoming_img_obj.view(num_img_obj, -1))            
            message_received_img_eve = self.fc_mp_receive_img_eve(message_incoming_img_eve.view(num_img_eve, -1))
            message_received_img_sce = self.fc_mp_receive_img_sce(message_incoming_img_sce.view(num_img_sce, -1))
            message_received_img_msk = self.fc_mp_receive_img_msk(message_incoming_img_msk.view(num_img_msk, -1))

            #print(f'{t}:mess:{message_received_img_msk}')

            '''
            self.debug_info[f'received_message_size_{t}'] = [
                torch.pow(message_received_ont_ent, 2).sum(1).mean(0),
                torch.pow(message_received_ont_pred, 2).sum(1).mean(0),
                torch.pow(message_received_img_ent, 2).sum(1).mean(0),
                torch.pow(message_received_img_pred, 2).sum(1).mean(0),
            ]
            '''

            z_img_obj = torch.sigmoid(self.fc_eq3_w_img_obj(message_received_img_obj) + self.fc_eq3_u_img_obj(nodes_img_obj))
            r_img_obj = torch.sigmoid(self.fc_eq4_w_img_obj(message_received_img_obj) + self.fc_eq4_u_img_obj(nodes_img_obj))
            h_img_obj = torch.tanh(self.fc_eq5_w_img_obj(message_received_img_obj) + self.fc_eq5_u_img_obj(r_img_obj * nodes_img_obj))
            nodes_img_obj_new = (1 - z_img_obj) * nodes_img_obj + z_img_obj * h_img_obj

            z_img_eve = torch.sigmoid(self.fc_eq3_w_img_eve(message_received_img_eve) + self.fc_eq3_u_img_eve(nodes_img_eve))
            r_img_eve = torch.sigmoid(self.fc_eq4_w_img_eve(message_received_img_eve) + self.fc_eq4_u_img_eve(nodes_img_eve))
            h_img_eve = torch.tanh(self.fc_eq5_w_img_eve(message_received_img_eve) + self.fc_eq5_u_img_eve(r_img_eve * nodes_img_eve))
            nodes_img_eve_new = (1 - z_img_eve) * nodes_img_eve + z_img_eve * h_img_eve

            z_img_sce = torch.sigmoid(self.fc_eq3_w_img_sce(message_received_img_sce) + self.fc_eq3_u_img_sce(nodes_img_sce))
            r_img_sce = torch.sigmoid(self.fc_eq4_w_img_sce(message_received_img_sce) + self.fc_eq4_u_img_sce(nodes_img_sce))
            h_img_sce = torch.tanh(self.fc_eq5_w_img_sce(message_received_img_sce) + self.fc_eq5_u_img_sce(r_img_sce * nodes_img_sce))
            nodes_img_sce_new = (1 - z_img_sce) * nodes_img_sce + z_img_sce * h_img_sce

            z_img_msk = torch.sigmoid(self.fc_eq3_w_img_msk(message_received_img_msk) + self.fc_eq3_u_img_msk(nodes_img_msk))
            r_img_msk = torch.sigmoid(self.fc_eq4_w_img_msk(message_received_img_msk) + self.fc_eq4_u_img_msk(nodes_img_msk))
            h_img_msk = torch.tanh(self.fc_eq5_w_img_msk(message_received_img_msk) + self.fc_eq5_u_img_msk(r_img_sce * nodes_img_msk))
            nodes_img_msk_new = (1 - z_img_msk) * nodes_img_msk + z_img_msk * h_img_msk

            #print(f'{t}   {z_img_msk}   {r_img_msk}   {h_img_msk}')

            relative_state_change_img_obj = torch.sum(torch.abs(nodes_img_obj_new - nodes_img_obj)) / torch.sum(torch.abs(nodes_img_obj))
            relative_state_change_img_eve = torch.sum(torch.abs(nodes_img_eve_new - nodes_img_eve)) / torch.sum(torch.abs(nodes_img_eve))
            relative_state_change_img_sce = torch.sum(torch.abs(nodes_img_sce_new - nodes_img_sce)) / torch.sum(torch.abs(nodes_img_sce))
            relative_state_change_img_msk = torch.sum(torch.abs(nodes_img_msk_new - nodes_img_msk)) / torch.sum(torch.abs(nodes_img_msk))

            self.debug_info[f'relative_state_change_{t}'] = [
                relative_state_change_img_obj, 
                relative_state_change_img_eve,
                relative_state_change_img_sce,
                relative_state_change_img_msk
            ]
        
            nodes_img_obj = nodes_img_obj_new
            nodes_img_eve = nodes_img_eve_new
            nodes_img_sce = nodes_img_sce_new
            nodes_img_msk = nodes_img_msk_new

            #print(f'{t}:ont_obj{nodes_ont_obj}')
            #print(f'{t}:ont_eve{nodes_ont_eve}')
            #print(f'{t}:ont_sce{nodes_ont_sce}')
            #print(f'{t}:img_obj{nodes_img_obj}')
            #print(f'{t}:img_eve {nodes_img_eve}')
            #print(f'{t}:img_sce{nodes_img_sce}')
            #print(f'{t}:img_msk {nodes_img_msk}')
                
            
            #print(nodes_img_obj.size(),nodes_ont_obj.size(),nodes_img_eve.size(),nodes_ont_eve.size(),global_feature_img.size())
            
            '''obj_weight=torch.max(obj_fg_cls_probs, dim=1).values / obj_fg_cls_probs.sum()
            obj_weighted_feat=self.fc_init_img_obj(nodes_feat[num_img_msk:,:] * mask_adj.view(-1, 1)) * obj_weight.view(-1, 1)
            #print(obj_weighted_feat.sum(dim=0, keepdim=True).size(),nodes_img_eve.size())
            nodes_img_eve = nodes_img_eve + obj_weighted_feat.sum(dim=0, keepdim=True)'''
            
            # 相似度loss
            eve_cls_logits = self.fc_output_proj_img_eve(nodes_img_eve)
            #eve_logits_steps.append(eve_cls_logits)
            # 更新img_eve到ont_eve的边
            if not self.multi_label:
                eve_fg_cls_probs = F.softmax(eve_cls_logits, dim=1)
            else:
                eve_fg_cls_probs = torch.sigmoid(eve_cls_logits)
            #print(f'eve:{edges_img2ont_eve}')

            #msk_cls_logits = self.fc_output_img_msk(nodes_img_msk)
            #print(f'eve:{eve_cls_logits}')
            #print(f'msk:{msk_cls_logits}')

            #nodes_ont_msk = nodes_ont_obj
            #print(f'nodes_img_msk_:{nodes_img_msk}')
            #nodes_ont_msk = nodes_ont_obj.clone()
            msk_cls_logits = self.fc_output_proj_img_msk(nodes_img_msk) 

        eps=0
        ops=None
        if self.privacy_score:
            eps, ops = self.PrivacyScore(proposal, eve_cls_logits, msk_cls_logits, mask_adj)

        return eve_cls_logits, msk_cls_logits, eps, ops

