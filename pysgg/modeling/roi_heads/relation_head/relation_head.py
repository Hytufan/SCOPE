# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    RelationProposalModel,
    filter_rel_pairs,
)
from pysgg.utils.visualize_graph import *
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .sampling import make_roi_relation_samp_processor
from ..attribute_head.roi_attribute_feature_extractors import (
    make_roi_attribute_feature_extractor,
)
from ..box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from pysgg.modeling.matcher import Matcher
from pysgg.structures.boxlist_ops import boxlist_iou
from .model_bgnn import BGNNContext
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.modeling.utils import cat
from .utils_relation import obj_prediction_nms
from .protected_predictor.protected_predictor import build_roi_protected_predictor
from .protected_predictor.kgevent import GGNNRelReason

def assign_isprotect_to_proposals(proposals, targets):
    proposal_matcher = Matcher(
            0.7,
            0.3,
            allow_low_quality_matches=False,
        )
    for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["is_protect"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        
        isprotect_per_image = matched_targets.get_field("is_protect").to(dtype=torch.int64)
        isprotect_per_image[matched_idxs < 0] = 0
        proposals[img_idx].add_field("is_protect", isprotect_per_image)
    return proposals

def update_predict_info(p):
    obj_pred_logits = p.get_field("predict_logits")
    boxes_per_cls = p.bbox.unsqueeze(1).expand(p.bbox.shape[0], obj_pred_logits.shape[-1],
                                                p.bbox.shape[-1]).contiguous()
    p.add_field("boxes_per_cls", boxes_per_cls)
    obj_pred_labels = obj_prediction_nms(boxes_per_cls, obj_pred_logits, nms_thresh=0.5)
    p.add_field("pred_labels", obj_pred_labels)

    obj_scores = torch.softmax(obj_pred_logits, 1).detach()
    obj_score_ind = torch.arange(obj_pred_logits.shape[0], device=obj_scores.device) * obj_pred_logits.shape[
        1] + obj_pred_labels
    obj_scores = obj_scores.view(-1)[obj_score_ind]
    p.add_field("pred_scores", obj_scores)
    return p

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()

        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        if not cfg.PROTECTED.TRAINING:
            self.union_feature_extractor = make_roi_relation_feature_extractor(
                cfg,
                in_channels,
            )
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(
                cfg, in_channels, half_out=True
            )
            self.att_feature_extractor = make_roi_attribute_feature_extractor(
                cfg, in_channels, half_out=True
            )
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            # the fix features head for extracting the instances ROI features for
            # obj detection
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
            if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
                feat_dim = self.box_feature_extractor.flatten_out_channels

        if cfg.PECONFIG.HIDESEEKER:
            self.protected_predictor=build_roi_protected_predictor(cfg, in_channels)
        elif cfg.PECONFIG.KG_EVENT:
            self.protected_predictor=GGNNRelReason(cfg)

        if not cfg.PECONFIG.TRAIN_LABEL and not cfg.PROTECTED.TRAINING:
            self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.rel_prop_on = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        self.object_cls_refine = cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        self.pass_obj_recls_loss = cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.use_protect_box = cfg.PECONFIG.USE_PROTECT_BOX
        self.get_avg_feature=cfg.PECONFIG.GET_AVG_FEATURE
        self.new_order=cfg.PECONFIG.NEW_ORDER
        self.train_label=cfg.PECONFIG.TRAIN_LABEL
        self.hideseeker=cfg.PECONFIG.HIDESEEKER
        self.protect_training=cfg.PROTECTED.TRAINING
        self.event_training=cfg.EVENT.TRAINING
        self.kgevent=cfg.PECONFIG.KG_EVENT

        if self.new_order:
            self.obj_context_layer = BGNNContext(
                cfg,
                feat_dim,
                hidden_dim=cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM,
                num_iter=cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
                pred_type='obj'
            )
            self.obj_classifier = build_classifier(cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        self.rel_pn_thres = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
        self.rel_pn_thres_for_test = torch.nn.Parameter(
            torch.Tensor(
                [
                    0.33,
                ]
            ),
            requires_grad=False,
        )
        self.rel_pn = None
        self.use_relness_ranking = False
        self.use_same_label_with_clser = False
        if self.rel_prop_on:
            if self.rel_prop_type == "rel_pn":
                self.rel_pn = RelationProposalModel(cfg)
                self.use_relness_ranking = (
                    cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_RELATEDNESS_FOR_PREDICTION_RANKING
                )
            if self.rel_prop_type == "pre_clser":
                self.use_same_label_with_clser == cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_SAME_LABEL_WITH_CLSER

    def forward(self, features, proposals, targets=None, protect_targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        roi_features = self.box_feature_extractor(features, proposals)
        if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
            roi_features = self.box_feature_extractor.flatten_roi_features(roi_features)

        if self.use_protect_box and 'is_protect' not in proposals[0].fields():
            if self.event_training or self.kgevent:
                proposals=assign_isprotect_to_proposals(proposals,protect_targets)
            else:
                proposals=assign_isprotect_to_proposals(proposals,targets)

        if self.hideseeker or self.kgevent:
            proposals, protect_loss, nodes_output=self.protected_predictor(proposals,protect_targets,roi_features)
            if self.protect_training:
                if self.training:
                    output_losses = dict()
                    #output_losses = dict(loss_protect_obj=protect_loss)
                    #if protect_loss[1]==0:
                        #output_losses = dict(loss_event=protect_loss[0])
                    #else:
                    output_losses = dict(loss_event=protect_loss[0],loss_obj=protect_loss[1])
                    #for i in range(len(protect_loss)):
                        #output_losses[f'loss_event_{i}']=protect_loss[i]

                    output_losses_checked = {}
                    for key in output_losses.keys():
                        if output_losses[key] is not None:
                            if output_losses[key].grad_fn is not None:
                                output_losses_checked[key] = output_losses[key]
                    output_losses = output_losses_checked
                    return roi_features, proposals, output_losses
                else:
                    return roi_features, proposals, {}

        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    (
                        proposals,
                        rel_labels,
                        rel_pair_idxs,
                        gt_rel_binarys_matrix,
                    ) = self.samp_processor.gtbox_relsample(proposals, targets)

                    rel_labels_all = rel_labels 
                else:
                    (
                        proposals,
                        rel_labels,
                        rel_labels_all,
                        rel_pair_idxs,
                        gt_rel_binarys_matrix,
                    ) = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_labels_all, gt_rel_binarys_matrix = None, None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(
                features[0].device, proposals
            )

        if self.mode == "predcls":
            # overload the pred logits by the gt label
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))
                proposal.add_field("pred_labels", obj_labels.to(device))

        # use box_head to extract features that will be fed to the later predictor processing
        #roi_features = self.box_feature_extractor(features, proposals)
        #if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
        #    roi_features = self.box_feature_extractor.flatten_roi_features(roi_features)

        if self.get_avg_feature:
            avg_file='/datapool/workspace/huangyunyi/code/SceneGraph/PySGG/checkpoints/avg_feature_4.pt'
            #device = features[0].device
            if os.path.exists(avg_file):
                feature_dict = torch.load(avg_file,map_location='cpu')
            else:
                feature_dict={}
            roi_tar_features=self.box_feature_extractor(features, targets)
            labels=[]
            for target in targets:
                label=target.get_field("labels").detach().cpu().numpy().tolist()
                labels=labels+label
            for label,roi_tar_feature in zip(labels,roi_tar_features):
                roi_tar_feature=roi_tar_feature.detach().cpu()
                if label in feature_dict:
                    old_feat=feature_dict[label]["feat"]
                    num=feature_dict[label]["num"]
                    new_feat=((num * old_feat) + roi_tar_feature) / (num+1)
                    feature_dict[label]["num"]+=1
                    feature_dict[label]["feat"]=new_feat
                else:
                    feature_dict[label]={}
                    feature_dict[label]["num"]=1
                    feature_dict[label]["feat"]=roi_tar_feature
            if os.path.exists(avg_file):        
                os.remove(avg_file)
            torch.save(feature_dict, avg_file)

            return roi_features,proposals,{}

        if self.new_order:
            if self.use_union_box:
                union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
            else:
                union_features = None
            obj_feats, _, _, _ = self.obj_context_layer(
                roi_features, union_features, proposals, rel_pair_idxs, gt_rel_binarys_matrix, logger
            )

            refined_obj_logits = self.obj_classifier(obj_feats)
            obj_refine_logits_ori = [prop.get_field("predict_logits") for prop in proposals]
            obj_fuse_logits=[]
            for ori_logits,ref_logits,proposal in zip(obj_refine_logits_ori,refined_obj_logits,proposals):
                is_protect=proposal.get_field("is_protect")
                fuse_logits=torch.where(is_protect.unsqueeze(1) == 1, ref_logits, ori_logits)
                obj_fuse_logits.append(fuse_logits)

            for i in range(len(proposals)):
                proposals[i].add_field("predict_logits",obj_fuse_logits[i])
                proposals[i]=update_predict_info(proposals[i])

            if self.train_label:
                if self.training:
                    num_objs = [len(b) for b in proposals]
                    obj_refine_logits = refined_obj_logits.split(num_objs, dim=0)          

                    loss_refine = self.loss_evaluator.label_loss(
                        proposals, obj_refine_logits
                    )       
                    output_losses = dict(loss_refine_obj=loss_refine)
                    return roi_features, proposals, output_losses
                else:
                    return roi_features, proposals, {}


        rel_pn_loss = None
        relness_matrix = None
        if self.rel_prop_on:
            fg_pair_matrixs = None
            gt_rel_binarys_matrix = None

            if targets is not None:
                fg_pair_matrixs, gt_rel_binarys_matrix = gt_rel_proposal_matching(
                    proposals,
                    targets,
                    self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                    self.cfg.TEST.RELATION.REQUIRE_OVERLAP,
                )
                gt_rel_binarys_matrix = [each.float().cuda() for each in gt_rel_binarys_matrix]


            if self.rel_prop_type == "rel_pn":
                relness_matrix, rel_pn_loss = self.rel_pn(
                    proposals,
                    roi_features,
                    rel_pair_idxs,
                    rel_labels,
                    fg_pair_matrixs,
                    gt_rel_binarys_matrix,
                )

                rel_pair_idxs, rel_labels = filter_rel_pairs(
                    relness_matrix, rel_pair_idxs, rel_labels
                )
                for enti_prop, rel_mat in zip(proposals, relness_matrix):
                    enti_prop.add_field('relness_mat', rel_mat.unsqueeze(-1)) 

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        rel_pn_labels = rel_labels
        if not self.use_same_label_with_clser:
            rel_pn_labels = rel_labels_all


        obj_refine_logits, relation_logits, add_losses, bgnn_feats = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_pn_labels,
            gt_rel_binarys_matrix,
            roi_features,
            union_features,
            nodes_output,
            logger,
        )

        # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        if not self.training:
            # re-NMS on refined object prediction logits
            if not self.object_cls_refine:
                # if don't use object classification refine, we just use the initial logits
                obj_refine_logits = [prop.get_field("predict_logits") for prop in proposals]
            if self.use_protect_box and self.object_cls_refine:
                #proposals=assign_isprotect_to_proposals(proposals,targets)
                obj_refine_logits_ori = [prop.get_field("predict_logits") for prop in proposals]
                obj_fuse_logits=[]
                for ori_logits,ref_logits,proposal in zip(obj_refine_logits_ori,obj_refine_logits,proposals):
                    is_protect=proposal.get_field("is_protect")
                    fuse_logits=torch.where(is_protect.unsqueeze(1) == 1, ref_logits, ori_logits)
                    obj_fuse_logits.append(fuse_logits)
                obj_refine_logits=obj_fuse_logits

            result = self.post_processor(
                (relation_logits, obj_refine_logits), rel_pair_idxs, proposals
            )

            if self.event_training:
                return nodes_output,result,bgnn_feats

            return roi_features, result, {}
        
        if self.new_order:
            num_objs = [len(b) for b in proposals]
            obj_refine_logits = refined_obj_logits.split(num_objs, dim=0)          

        loss_relation, loss_refine = self.loss_evaluator(
            proposals, rel_labels, relation_logits, obj_refine_logits
        )

        output_losses = dict()
        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(
                loss_rel=loss_relation,
                loss_refine_obj=loss_refine[0],
                loss_refine_att=loss_refine[1],
            )
        else:
            if self.pass_obj_recls_loss or self.new_order:
                output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
            else:
                output_losses = dict(loss_rel=loss_relation)

        if rel_pn_loss is not None:
            output_losses["loss_relatedness"] = rel_pn_loss

        output_losses.update(add_losses)
        output_losses_checked = {}
        if self.training:
            for key in output_losses.keys():
                if output_losses[key] is not None:
                    if output_losses[key].grad_fn is not None:
                        output_losses_checked[key] = output_losses[key]
        output_losses = output_losses_checked
        #print(output_losses)
        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
