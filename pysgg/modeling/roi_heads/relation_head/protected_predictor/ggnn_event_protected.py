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
import pandas as pd

from .ggnn_pa import GGNN
from .relation import RelationLearner
from torch_geometric.explain import GNNExplainer, Explainer
from lime import lime_image
from skimage.segmentation import slic
from PIL import Image
import shap
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

import os

def ensure_directory_exists(filename):
    # 获取文件所在的目录路径
    directory = os.path.dirname(filename)
    
    # 如果目录不为空且不存在，则创建目录
    if directory and not os.path.exists(directory):
        os.makedirs(directory)



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
        self.obj_on=cfg.PROTECTED.OBJ
        self.event_on=cfg.PROTECTED.EVENT
        self.multi_label=cfg.PROTECTED.MULTI_LABEL


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
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.explainer=None
        self.explainer_type=cfg.KGEVENT.EXPLAIN
        if cfg.KGEVENT.EXPLAIN=='GNNExplainer':
            self.explainer = Explainer(
            model=self.ggnn,
            algorithm=GNNExplainer(epochs=100,lr=0.1),
            explanation_type='phenomenon',
            node_mask_type='object',
            edge_mask_type=None,
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
        )
        elif 'LIME' in cfg.KGEVENT.EXPLAIN:
            self.explainer = lime_image.LimeImageExplainer()
            
    def model_predict(self, image, annotation, proposal, protect_target, scene_info, adj_matrix):
        
        image_np = np.array(image)  # 确保是 NumPy 数组
        image_tensor = torch.tensor(image_np, dtype=torch.float32) / 255.0  # 转换为 [0, 1] 的浮动值
        #print(f'2 {image_tensor.shape}') 
        image = image_tensor.squeeze(0).permute(2, 0, 1).to(self.device)  # 变为 C x H x W 格式
        #print(f'3 {image.shape}')
        # 假设 self.ggnn() 返回的是 logits (el, ol, eps, ops)
        el, ol, eps, ops = self.ggnn(annotation, proposal, protect_target, scene_info, adj_matrix, image)
        
        # 返回目标分类的 logits
        return el.cpu().detach().numpy()

    # 使用闭包将额外的参数传递给 `model_predict`
    def create_predict_function(self, annotation, proposal, protect_target, scene_info, adj_matrix):
        def predict(images):
            """
            这个闭包接受一个 `images` 输入，并使用外部的额外上下文参数进行预测
            """
            return self.model_predict(images, annotation, proposal, protect_target, scene_info, adj_matrix)
        return predict
    
    def create_segment_function(self,bboxes):
        def segment_bbox(image):
            return bbox_to_slic_like(image, bboxes)
        return segment_bbox

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
            proposal.add_field("pred_scene",scene_info['labels'][0])
            
            roi_feature=roi_features[idx:idx+len(proposal),:]
            roi_feature_protected=roi_features_protected[idx_p:idx_p+len(protect_target),:]
            idx+=len(proposal)
            idx_p+=len(protect_target)

            image=protect_target.get_field('image')
            masked_info, others_info = self.get_objects_info(image, proposal, protect_target, roi_feature, roi_feature_protected)
            adj_matrix, annotation = self.relation_learner(self.device, image, masked_info, others_info)
            #print("info:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
            #el, ol = self.ggnn(proposal, roi_feature, scene_info)
            el, ol, eps, ops = self.ggnn(annotation, proposal, protect_target, scene_info, adj_matrix, image)
            #print("ggnn:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
            if self.explainer_type=='GNNExplainer':
                #print(f'1 {eps}')
                explanation = self.explainer(annotation, None, protect_target=protect_target, target=el, proposal=proposal, scene_info=scene_info, adj_matrix=adj_matrix, image=image)
                proposal.add_field("explain",explanation['node_mask'].squeeze(dim=1).cpu().detach().numpy())
                torch.cuda.empty_cache()
                #print(explanation['node_mask'].squeeze(dim=1))
            elif self.explainer_type=='LIME':
                predict_function = self.create_predict_function(annotation, proposal, protect_target, scene_info, adj_matrix)
                image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换为 H x W x C
                image_np = (image_np * 255).astype(np.uint8)
                #print(f'1 {image_np.shape}')
                explanation = self.explainer.explain_instance(
                    image_np,  # 只传入一张图像，LIME 会处理批量的扰动
                    predict_function,  # 使用我们定义的适配器函数
                    top_labels=1,  # 返回前5个类别
                    hide_color=0,  # 设为0来隐藏背景区域
                    num_samples=1500,  # 扰动样本数
                    segmentation_fn=slic,  # 图像分割方法（这里使用 SLIC 超像素分割）
                    batch_size=1
                )

                # 获取解释的图像和掩膜
                '''temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],  # 显示最重要的标签
                    positive_only=True,  # 显示对决策有正面影响的区域
                    num_features=5,  # 显示最重要的5个区域
                    hide_rest=True  # 隐藏对决策无关的区域
                )'''
                
                weight_map = get_weight_map(explanation, label=explanation.top_labels[0])
                save_path=os.path.join('/workspace/huangyunyi/data/MyEventDataset/prs_0115/lime_mask_1500/',protect_target.get_field('file_name').replace('jpg','npy'))
                ensure_directory_exists(save_path)
                np.save(save_path, weight_map)
                
                '''temp_image = Image.fromarray((temp * 255).astype(np.uint8))  # 如果 temp 是浮动值，转换为 0-255 的整数
                save_path=os.path.join(protect_target.get_field('file_name').replace('/workspace/huangyunyi/data/MyEventDataset/original/','/workspace/huangyunyi/data/MyEventDataset/background_on/lime/'))
                #print(save_path)
                ensure_directory_exists(save_path)
                temp_image.save(save_path)'''
            elif self.explainer_type=='LIME_bbox':
                predict_function = self.create_predict_function(annotation, proposal, protect_target, scene_info, adj_matrix)
                image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换为 H x W x C
                image_np = (image_np * 255).astype(np.uint8)
                #print(f'1 {image_np.shape}')
                segment_bbox = self.create_segment_function(proposal.bbox.clone().cpu().detach())
                explanation = self.explainer.explain_instance(
                    image_np,  # 只传入一张图像，LIME 会处理批量的扰动
                    predict_function,  # 使用我们定义的适配器函数
                    top_labels=1,  # 返回前5个类别
                    hide_color=0,  # 设为0来隐藏背景区域
                    num_samples=1500,  # 扰动样本数
                    segmentation_fn=segment_bbox,  # 图像分割方法（这里使用 SLIC 超像素分割）
                    batch_size=1
                )
                weight_map = get_weight_map(explanation, label=explanation.top_labels[0])
                save_path=os.path.join('/workspace/huangyunyi/data/MyEventDataset/prs_0115/lime_bbox_1500/',protect_target.get_field('file_name').replace('jpg','npy'))
                ensure_directory_exists(save_path)
                np.save(save_path, weight_map)
            elif self.explainer_type=='shap':
                explainer = shap.KernelExplainer(self.ggnn, proposal)
                #shap_values = explainer.shap_values(input_data)
                
            event_logits.append(el)
            obj_logits_refined.append(ol)

            proposal.add_field("el",el.cpu().detach())
            proposal.add_field("ol",ol.cpu().detach())
            #proposal.add_field("ops",ops)
            proposal.add_field("eps",eps)
            proposal.add_field("ops",ops)
            proposal.add_field("filename",protect_target.get_field('file_name'))
            #print(ops,eps)
            #proposal.add_field("attention_scores",ops)

            pred_label = torch.argmax(el, dim=-1)
            proposal.add_field("pred_event_label",pred_label)

            if self.obj_on:
                pred_obj = torch.argmax(ol, dim=-1)
                pred_obj_list=pred_obj.tolist()
                pred_obj_labels=[self.class_map[pred_obj_index] for pred_obj_index in pred_obj_list]
                #proposal.add_field("pred_obj_label",torch.tensor([self.class_map[pred_obj]]))
                proposal.add_field("pred_obj_label",torch.tensor(pred_obj_labels))
            else:
                proposal.add_field("pred_obj_label",torch.tensor([0]))

            target_label=protect_target.get_field("event_label").unsqueeze(0)
            proposal.add_field("event_label",target_label)
            proposal.add_field("mask_bbox",protect_target.bbox)
            if pred_conf_all==None:
                pred_conf_all=el
                target_labels_all=target_label
                #el_conf_all=el_conf
                '''for i in range(len(el_steps)):
                    pred_conf_all_list.append(el_steps[i])'''
                if self.obj_on:
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
                if self.obj_on:
                    pred_obj_all=torch.cat((pred_obj_all,ol))
                    target_obj=protect_target.get_field("protect_labels")
                    proposal.add_field("protect_labels",target_obj)
                    target_obj_trans=torch.zeros_like(target_obj, dtype=torch.long)
                    for i in range(len(target_obj)):
                        target_obj_trans[i] = list(self.class_map).index(target_obj[i])
                    target_obj_all=torch.cat((target_obj_all,target_obj_trans))
            #print("event1:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        #print("event2:",datetime.datetime.now().strftime("%H:%M:%S.{:03d}".format(datetime.datetime.now().microsecond // 1000)))
        if self.event_on:
            if not self.multi_label:
                loss = self.criterion_loss(pred_conf_all, target_labels_all.view(-1).long())
            elif self.multi_label:
                loss = self.bce_loss(pred_conf_all, target_labels_all.float())
            loss=[loss] #event
        else:
            loss = [] #no event
        if self.obj_on:
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
            
        for proposal in proposals:
            for field in proposal.fields():
                field_data = proposal.get_field(field)

                # 如果是 Tensor，则执行 detach 操作
                if isinstance(field_data, torch.Tensor):
                    detached_data = field_data.detach().cpu()
                    proposal.add_field(field, detached_data)
                    
                # 如果字段数据是其他数据类型（如字典、列表），可以递归处理
                elif isinstance(field_data, dict):
                    for key, value in field_data.items():
                        if isinstance(value, torch.Tensor):
                            field_data[key] = value.detach().cpu()
                    proposal.add_field(field, field_data)

                elif isinstance(field_data, list):
                    field_data = [
                        item.detach().cpu() if isinstance(item, torch.Tensor) else item
                        for item in field_data
                    ]
                    proposal.add_field(field, field_data)
                
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

def get_weight_map(explanation, label):
    """
    Creates a NumPy array where each pixel's value is the weight of its superpixel.

    Args:
        explanation: LimeImageExplanation object.
        label: The label for which to extract the weight map.

    Returns:
        A 2D NumPy array where each pixel's value corresponds to its superpixel's weight.
    """
    if label not in explanation.local_exp:
        raise KeyError(f"Label {label} not found in explanation.")

    # Get the superpixel weights
    weights = dict(explanation.local_exp[label])
    #print('num:',len(weights),'                    ')

    # Get the segments
    segments = explanation.segments

    # Create a 2D array initialized to 0
    weight_map = np.zeros_like(segments, dtype=np.float32)

    # Map weights to the corresponding superpixels
    for superpixel, weight in weights.items():
        weight_map[segments == superpixel] = weight

    return weight_map
def bbox_to_slic_like(image, bboxes):
    """
    Generate a SLIC-like segmentation map using bounding boxes (bboxes).

    Args:
        image (np.ndarray): Input image as a 2D or 3D array (H, W, C).
        bboxes (list of lists): List of bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: SLIC-like segmentation map where each pixel has a label corresponding to a bbox.
    """
    height, width = image.shape[:2]
    label_map = np.full((height, width), -1, dtype=int)  # Initialize with -1 for unassigned pixels

    # Compute areas of bboxes
    bbox_areas = [
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if bbox[2] > bbox[0] and bbox[3] > bbox[1] else float('inf')
        for bbox in bboxes
    ]

    # Sort bboxes by area (smallest first)
    sorted_indices = np.argsort(bbox_areas)
    sorted_bboxes = [bboxes[i] for i in sorted_indices]

    for label, bbox in enumerate(sorted_bboxes):
        x_min, y_min, x_max, y_max = bbox

        # Ensure bbox coordinates are valid
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width, int(x_max))
        y_max = min(height, int(y_max))

        # Assign labels to the bbox region, respecting existing smaller-area labels
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if label_map[y, x] == -1:  # Unassigned pixel
                    label_map[y, x] = label

    return label_map