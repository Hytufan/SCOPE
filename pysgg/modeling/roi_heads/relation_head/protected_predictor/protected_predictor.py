import torch
import cv2
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from .label_smooth import LabelSmoothSoftmaxCE
from pysgg.modeling.utils import cat
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier

from .relation_learner import RelationLearner
from .ggnn import GGNN
from pysgg.modeling.make_layers import make_fc

class ROIProtectedPredictor(torch.nn.Module):

    def __init__(self, cfg, in_channels):
        super(ROIProtectedPredictor, self).__init__()

        self.init_weight=True
        self.is_pred=False
        self.device=None
        self.cfg = cfg.clone()
        self.num_classes = cfg.PROTECTED.NUM_CLASSES
        self.class_map=cfg.PROTECTED.PROTECT_CLASSES
        
        self.relation_learner = RelationLearner(cfg)
        self.ggnn = GGNN(cfg.PROTECTED.GGNN_D_STATE, cfg.PROTECTED.GGNN_D_ANNOTATION, 
                        cfg.PROTECTED.GGNN_N_NODES, cfg.PROTECTED.GGNN_N_EDGE_TYPES, cfg.PROTECTED.GGNN_N_STEPS)
        self.dropout = nn.Dropout()
        self.leakyrelu = nn.LeakyReLU()
        
        #self.fc1 = nn.Linear(self.ggnn.state_dim, self.num_classes)
        #self.classifier = nn.Softmax(dim=-1)
        self.classifier=build_classifier(self.ggnn.state_dim, self.num_classes)
        self.fc2=make_fc(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM,cfg.PROTECTED.FEATURE_DIM)

        if self.init_weight:
            self._initialize_weights()

        self.scene_recognizor = models.__dict__['resnet50'](num_classes=365)
        #scene_state_dict = torch.load('./models/resnet50_places365.pt', map_location=self.device)
        #self.scene_recognizor.load_state_dict(scene_state_dict)

        self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self,proposals,protect_targets,roi_features):
        self.device=roi_features.device

        new_proposals=[]
        nodes_outputs=[]
        pred_conf_all=None
        target_labels_trans_all=None
        idx=0
        for proposal, protect_target in zip(proposals,protect_targets):
        #proposal=proposal[0]
        #protect_target=protect_target[0]
            roi_feature=roi_features[idx:idx+len(proposal),:]
            idx+=len(proposal)

            #for proposal, protect_target in zip(proposals,protect_targets):
            roi_feature=F.relu(self.fc2(roi_feature))
            masked_info, others_info = self.get_objects_info(proposal,protect_target,roi_feature)
            image=protect_target.get_field('image')
            scene_info = self.get_scene_info(image)

            if len(masked_info):
                pred_bbox = torch.stack([m['box'] for m in masked_info]).unsqueeze(0)
            else:
                pred_bbox = torch.tensor([[[0, 0, 0, 0]]], device=self.device)
            
            n_masked = len(masked_info)
            n_other_objects = len(others_info)

            batch_size = 1
            adj_matrix, annotation = self.relation_learner(image, masked_info, others_info, scene_info)
            padding = torch.zeros((batch_size, self.cfg.PROTECTED.GGNN_N_NODES, self.cfg.PROTECTED.GGNN_D_STATE - self.cfg.PROTECTED.GGNN_D_ANNOTATION), device=self.device)
            adj_matrix, annotation = adj_matrix.unsqueeze(0), annotation.unsqueeze(0)
            init_input = torch.cat((annotation, padding), 2)
            ggnn_output, nodes_output = self.ggnn(init_input, annotation, adj_matrix, n_masked)

            #x = self.fc1(ggnn_output)
            #output = self.classifier(x)
            output = self.classifier(ggnn_output)
            prediction = torch.cat((pred_bbox, output), dim=-1)

            pred_boxes, pred_conf = prediction[0, :, :4], prediction[0, :, 4:]
            pred_label = torch.argmax(pred_conf, dim=-1)

            target_labels=protect_target.get_field('protect_labels')
            target_labels_trans=torch.zeros_like(target_labels, dtype=torch.long)
            for i in range(len(target_labels)):
                target_labels_trans[i] = list(self.class_map).index(target_labels[i])

            #loss=self.celoss(prediction,target_labels_trans)
            #refine_obj_logits = cat(pred_conf, dim=0)
            #fg_labels = cat([target_labels_trans], dim=0)
            #print(pred_label,target_labels_trans)

            if pred_conf_all==None:
                pred_conf_all=pred_conf
                target_labels_trans_all=target_labels_trans
            else:
                pred_conf_all=torch.cat((pred_conf_all,pred_conf))
                target_labels_trans_all=torch.cat((target_labels_trans_all,target_labels_trans))
            #loss = self.criterion_loss(pred_conf, target_labels_trans.long())

            is_protect=proposal.get_field('is_protect')
            pred_labels=proposal.get_field('pred_labels')
            predict_logits=proposal.get_field('predict_logits')
            pred_label=torch.tensor([self.class_map[pred_label]]).to(self.device)
            
            pred_labels[is_protect==1]=pred_label
            new_logits = torch.full((predict_logits.size(-1),), float('-1e9')).to(self.device)
            new_logits[self.class_map] = pred_conf
            predict_logits[is_protect==1]=new_logits
            proposal.add_field('pred_labels',pred_labels)
            proposal.add_field('predict_logits',predict_logits)

            nodes_output=torch.squeeze(nodes_output, dim=0)

            new_proposals.append(proposal)
            nodes_outputs.append(nodes_output)
        loss = self.criterion_loss(pred_conf_all, target_labels_trans_all.long())
        return new_proposals, loss, nodes_outputs
    
    def get_objects_info(self, proposal, protect_target, roi_feature):
        masked_info = []
        #c, h, w = x.shape
        if self.is_pred:
            self.is_pred=True
            '''with torch.no_grad():
                output = self.masked_detector(x.unsqueeze(0))
                masked_output = output[0][0].cpu().detach().numpy() * 255
            mask = np.uint8(masked_output)
            _, mask = cv2.threshold(masked_output, 128, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, None, iterations=2)
            # cv2.imwrite('mask.jpg', mask)
            mask = np.uint8(mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = []
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                areas.append(area)
            sorted_areas = sorted(areas, reverse=True)

            std = np.std(sorted_areas)
            mean = np.mean(sorted_areas)
            max_areas = np.array(sorted_areas)[np.where(sorted_areas >= mean + std)]
            masked_inds = []
            for area in max_areas:
                indices = np.argwhere(areas == area).reshape(-1)
                masked_inds.extend(indices.tolist())
            # print(areas, masked_inds)
            black = np.zeros([h, w, 3], np.float32)
            masked_boxes = []
            for i in masked_inds:
                masked = {}
                cont = cv2.drawContours(black, contours, i, (255, 255, 255), thickness=-1)
                refi_mask = cv2.cvtColor(cont, cv2.COLOR_BGR2GRAY)
                _, refi_mask = cv2.threshold(refi_mask, 128, 255, cv2.THRESH_BINARY)
                mask_pixels = np.transpose(np.argwhere(refi_mask > 0.5), axes=(1, 0))
                top, left = np.min(mask_pixels, axis=1)
                bottom, right = np.max(mask_pixels, axis=1)
            
                masked['box'] = torch.tensor([left, top, right, bottom])
                masked_boxes.append(masked['box'])
                masked_info.append(masked)
                masked_boxes = torch.tensor(masked_boxes).to(self.device)'''
        else:
            masked_boxes=protect_target.bbox.detach()
            for i in range(len(masked_boxes)):
                masked = {}
                masked['box'] = torch.tensor(masked_boxes[i])
                x1, y1, x2, y2 = masked_boxes[i]
                masked['center'] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
                masked['area'] = torch.tensor((x2 - x1) * (y2 - y1))
                masked_info.append(masked)
        
        n_masked_objects = len(masked_info)
        others_info = []
        n_other_objects = self.cfg.PROTECTED.GGNN_N_NODES - n_masked_objects - 1
        
        boxes = proposal.bbox.detach()[:n_other_objects]
        labels = proposal.get_field('pred_labels').detach()[:n_other_objects]
        scores = proposal.get_field('pred_scores').detach()[:n_other_objects]
        roi_feats = roi_feature.detach()[:n_other_objects]
        mask_feats = roi_feature[proposal.get_field('is_protect')==1].detach()[:n_masked_objects]
        if mask_feats.size()[0]==0:
            mask_feats = torch.zeros((n_masked_objects,roi_feature.size()[-1]), device=self.device)

        # print("get object package:\t", datetime.datetime.now())
        # object_preds = pickle.load( \
        #         open(os.path.join(dataset_dir, 'm_objs', idx + ".pkl"), 'rb'))
        # roi_feats = object_preds['roi_feats'].detach()
        # mask_feats = object_preds['mask_feats'].detach()
        # boxes = object_preds['boxes'].detach()
        # labels = object_preds['labels'].detach()
        # scores = object_preds['scores'].detach()
        
        for i in range(n_other_objects):
            if i>=len(boxes):
                obj = {}
                obj['box'] = torch.zeros(4).to(self.device)
                x1, y1, x2, y2 = obj['box']
                obj['score'] = torch.tensor(0).to(self.device)
                obj['label'] = torch.tensor(0).to(self.device)
                obj['feat'] = torch.zeros(self.cfg.PROTECTED.FEATURE_DIM).to(self.device)
                obj['center'] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
                obj['area'] = torch.tensor((x2 - x1) * (y2 - y1))
                others_info.append(obj)
                continue

            obj = {}
            obj['box'] = boxes[i]
            x1, y1, x2, y2 = boxes[i]
            obj['score'] = scores[i]
            obj['label'] = labels[i]
            obj['feat'] = roi_feats[i]
            obj['center'] = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
            obj['area'] = torch.tensor((x2 - x1) * (y2 - y1))
            others_info.append(obj)

        for i in range(n_masked_objects):
            masked_info[i]['feat'] = mask_feats[i]

        return masked_info, others_info
    
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
        scene_info['scores'] = torch.tensor([scores[idx].item() for idx in indices[0][:5]])
        
        # get scene package:
        # idx = str(int(idx)).rjust(12, '0')
        # scene_preds = pickle.load( \
        #         open(os.path.join(dataset_dir, 'scenes', idx + ".pkl"), 'rb'))
        # scene_info['feat'] = scene_preds['feat'].detach()
        # scene_info['labels'] = scene_preds['labels']
        # scene_info['scores'] = np.array(scene_preds['scores'])
    
        return scene_info

    def celoss(self, predictions, target_labels):
        ce = nn.CrossEntropyLoss(reduction='sum')
        lmce = LabelSmoothSoftmaxCE(reduction='none')
        # print("celoss | preds, targets:", predictions.shape, targets.shape) # predicitions: [N, 8]; targets: [N, 5]
        pred_boxes, pred_conf = predictions[0, :, :4], predictions[0, :, 4:]
        # target_boxes, target_labels = targets[:, :4], targets[:, 4]
        n_preds, n_targets = pred_conf.shape[0], target_labels.shape[0]
        # print("celoss | n_pred, n_targets:", n_preds, n_targets)
        # print(target_labels.dtype)
        if n_preds <= n_targets:
            loss = lmce(pred_conf, target_labels[:n_preds])
        else:
            loss = lmce(pred_conf, target_labels[:n_targets])
        # print(loss.item())
        return loss

    def accuracy(self, predictions, target_labels):
        # print("prediction, label:", predictions ,targets)
        pred_boxes, pred_conf = predictions[:, :4], predictions[:, 4:]
        # target_boxes, target_labels = targets[:, :4], targets[:, 4]
        n_preds, n_targets = pred_conf.shape[0], target_labels.shape[0]
        pred_labels = torch.argmax(pred_conf, dim=-1)
        # print("acc | pred_labels, target_labels:", pred_labels.shape, target_labels.shape)
        # print(pred_labels, labels)
        acc_num = (pred_labels == target_labels).sum()
        # if acc_num == 0:
        #     _, indices = torch.topk(predictions, 1, dim=-1, largest=True)
        #     print(indices)
        #     indices = indices.tolist()[0]
        #     print(indices)
        #     # print(indices)
        #     top1_preds = [list(cfg.pri_class_id_map)[i] for i in indices]
        #     groundtruth = cfg.pri_class_id_map[list(cfg.pri_class_id_map)[labels.tolist()[0]]]
        # print("acc num:", acc_num)
        return acc_num

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


def build_roi_protected_predictor(cfg, in_channels):
    return ROIProtectedPredictor(cfg, in_channels)