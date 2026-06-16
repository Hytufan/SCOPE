import json
import json
import os
import pickle

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from pysgg.data.datasets.evaluation.coco.coco_eval import COCOResults
from pysgg.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, \
    SGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGStagewiseRecall, SGNGMeanRecall
from pysgg.data.datasets.visual_genome import HEAD, TAIL, BODY
from pysgg.modeling.matcher import Matcher
from pysgg.structures.boxlist_ops import boxlist_iou
from pysgg.structures.bounding_box import BoxList
from sklearn.metrics import average_precision_score

eval_times = 0


def do_wider_evaluation(
        cfg,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
):
    # get zeroshot triplet
    #zeroshot_triplet = torch.load("pysgg/data/datasets/evaluation/vg/zeroshot_triplet.pytorch",
                                  #map_location=torch.device("cpu")).long().numpy()
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    #groundtruths = []
    #protect_targets=[]
    for image_id, prediction in enumerate(predictions):
        #protect_target=BoxList(prediction.get_field('mask_bbox'),prediction.size,prediction.mode)

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))

        #protect_target = protect_target.resize((image_width, image_height))
        #protect_targets.append(protect_target)

    # explain
    '''explain_info={}
    for image_id, prediction in enumerate(predictions):
        filename=dataset.filenames[image_id]
        mask_bbox=predictions[image_id].get_field('mask_bbox')
        all_bbox=np.concatenate((mask_bbox, predictions[image_id].bbox), axis=0).tolist()
        explain=predictions[image_id].get_field('explain').numpy().tolist()
        event=predictions[image_id].get_field('event_label').item()
        explain_info[filename]={"event":event,"explain":explain,"bboxes":all_bbox}
    with open('/workspace/huangyunyi/data/MyEventDataset/explain_result.json', 'w') as json_file:
        json.dump(explain_info, json_file)'''

    # 读取 JSON 文件
    '''with open('/workspace/huangyunyi/data/MyEventDataset/explain_result_label.json', 'r') as json_file:
        explain_info = json.load(json_file)

    for image_id, prediction in enumerate(predictions):
        filename=dataset.filenames[image_id]
        mask_bbox=protect_targets[image_id].bbox
        all_bbox=np.concatenate((mask_bbox, predictions[image_id].bbox), axis=0).tolist()
        explain_info[filename]["bboxes"]=all_bbox
        #explain_info[filename]["label"] = np.concatenate((prediction.get_field('pred_obj_label'), prediction.get_field('pred_labels')), axis=0).tolist()

    with open('/workspace/huangyunyi/data/MyEventDataset/explain_result_label_new.json', 'w') as json_file:
        json.dump(explain_info, json_file)'''

    #scene
    '''scene_info={}
    for image_id, prediction in enumerate(predictions):
        filename=dataset.filenames[image_id]
        #print(prediction.get_field('pred_scene').item())
        scene_info[filename]=prediction.get_field('pred_scene').item()
    with open('/workspace/huangyunyi/data/MyEventDataset/scene_info.json', 'w') as json_file:
        json.dump(scene_info, json_file)'''


    #object
    '''obj_info={}
    for image_id, prediction in enumerate(predictions):
        filename=dataset.filenames[image_id]
        labels=prediction.get_field("pred_labels").numpy().tolist()
        scores=prediction.get_field("pred_scores").numpy().tolist()
        bboxes=prediction.bbox.numpy().tolist()
        obj_info[filename]={"labels":labels,"scores":scores,"bboxes":bboxes}'''

    '''obj_info = {}

    for image_id, prediction in enumerate(predictions):
        filename = dataset.filenames[image_id]
        labels = prediction.get_field("pred_labels").numpy().tolist()
        scores = prediction.get_field("pred_scores").numpy().tolist()
        bboxes = prediction.bbox.numpy().tolist()

        # 筛选出得分大于0.5的标签、得分和边界框，并按照得分进行排序
        filtered_info = [
            (label, score, bbox) 
            for label, score, bbox in zip(labels, scores, bboxes) 
            if score > 0.5
        ]
        filtered_info.sort(key=lambda x: x[1], reverse=True)  # 按得分降序排序

        filtered_labels = [info[0] for info in filtered_info]
        filtered_scores = [info[1] for info in filtered_info]
        filtered_bboxes = [info[2] for info in filtered_info]

        # 将筛选后的信息存储到字典中
        obj_info[filename] = {"labels": filtered_labels, "scores": filtered_scores, "bboxes": filtered_bboxes}
    with open('/workspace/huangyunyi/data/MyEventDataset/obj_info_5.json', 'w') as json_file:
        json.dump(obj_info, json_file)'''

        
        #gt = dataset.get_groundtruth(image_id, evaluation=True)
        #groundtruths.append(gt)
    #save_output(output_folder, predictions, dataset)
    #return predictions
    avg_metrics = 0
    result_str = '\n' + '=' * 100 + '\n'

    result_dict = {}
    result_dict_list_to_log = []

    if "bbox" in iou_types:
        '''# create a Coco-like object that we can use to evaluate det#ection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist()  # integer
            boxes = gt.bbox.tolist()  # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
            ],
            'annotations': anns,
        }
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id] * len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
            )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)

        # logger.info("Evaluating bbox proposals")
        # areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        # res = COCOResults("box_proposal")
        # for limit in [100, 1000]:
        #     for area, suffix in areas.items():
        #         stats = evaluate_box_proposals(
        #             predictions, dataset, area=area, limit=limit
        #         )
        #         key = "AR{}@{:d}".format(suffix, limit)
        #         res.results["box_proposal"][key] = stats["ar"].item()
        # logger.info(res)
        # if output_folder:
        #     torch.save(res, os.path.join(output_folder, "box_proposals.pth"))

        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_res = COCOResults('bbox')
        coco_res.update(coco_eval)
        mAp = coco_eval.stats[1]

        def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
            p = coco_eval.params

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            if maxDets == -1:
                max_range_i = np.argmax(p.maxDets)
                mind = [max_range_i, ]
            else:
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if eval_type == 'precision':
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            elif eval_type == 'recall':
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            else:
                raise ValueError("Invalid eval metrics")
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return p.maxDets[mind[-1]], mean_s

        coco_res_to_save = {}
        for key, value in coco_res.results.items():
            for evl_name, eval_val in value.items():
                coco_res_to_save[f"{key}/{evl_name}"] = eval_val
        result_dict_list_to_log.append(coco_res_to_save)

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, 'recall')
        result_str += '=' * 100 + '\n'
        avg_metrics = mAp
        logger.info(result_str)
        result_str = '\n'
        logger.info("box evaluation done!")'''

        '''test_mode=True
        if test_mode:
            return '''
        label_eval=cfg.PROTECTED.OBJ
        if label_eval:
            logger.info("label evaluation start!")

            proposal_matcher = Matcher(
                cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
                allow_low_quality_matches=False,
            )

            def assign_label_to_proposals(proposals, targets):
                for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
                    match_quality_matrix = boxlist_iou(target, proposal)
                    matched_idxs = proposal_matcher(match_quality_matrix)
                    # Fast RCNN only need "labels" field for selecting the targets
                    target = target.copy_with_fields(["protect_labels","is_protect"])
                    matched_targets = target[matched_idxs.clamp(min=0)]
                    
                    labels_per_image = matched_targets.get_field("protect_labels").to(dtype=torch.int64)
                    #attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)
                    isprotect_per_image = matched_targets.get_field("is_protect").to(dtype=torch.int64)

                    labels_per_image[matched_idxs < 0] = 0
                    #attris_per_image[matched_idxs < 0, :] = 0
                    isprotect_per_image[matched_idxs < 0] = 0
                    proposals[img_idx].add_field("labels", labels_per_image)
                    #proposals[img_idx].add_field("attributes", attris_per_image)
                    proposals[img_idx].add_field("is_protect", isprotect_per_image)
                return proposals 

            gts=[]
            for image_id, prediction in enumerate(predictions):
                gt=dataset.get_protect_target(image_id,prediction.size)
                #gt=dataset.get_groundtruth(image_id, evaluation=True)
                #is_protect_list=dataset.get_is_protect(gt, pro_gt)
                #gt.add_field("is_protect",is_protect_list)
                gts.append(gt)
            predictions=assign_label_to_proposals(predictions,gts)
            pred_pro_dict={}
            pro_all_dict={}
            pred_label_dict={}
            for image_id, prediction in enumerate(predictions):
                pro_gt=dataset.get_protect_target(image_id,prediction.size)
                pro_label=pro_gt.get_field("protect_labels").detach().cpu().numpy()[0]
                is_protect_list=prediction.get_field("is_protect").detach().cpu().numpy()
                pred_labels=prediction.get_field("pred_labels").detach().cpu().numpy()

                if pro_label in pro_all_dict:
                    pro_all_dict[pro_label] += 1
                else:
                    pro_all_dict[pro_label] = 1
                    pred_pro_dict[pro_label] = 0

                pred_pro=pred_labels[is_protect_list==1]
                if pro_label in pred_pro:
                    if pro_label in pred_pro_dict:
                        pred_pro_dict[pro_label] += 1
                for pl in pred_pro:
                    if pl in pred_label_dict:
                        pred_label_dict[pl] += 1
                    else:
                        pred_label_dict[pl] = 1
            
            acc_all={}
            #print(pred_pro_dict,pro_all_dict)
            result_str+="pred_pro_dict:"+str(pred_pro_dict)+'\n'
            result_str+="pro_all_dict:"+str(pro_all_dict)+'\n'
            result_str+="pred_label_dict:"+str(pred_label_dict)+'\n'

            for key in pro_all_dict:
                acc_all[key]=pred_pro_dict[key]/pro_all_dict[key]
            acc_pro=sum([pred_pro_dict[key] for key in pred_pro_dict if key in pro_all_dict]) / sum(pro_all_dict.values())

            result_str+="pro_acc_all_classes:"+str(acc_all)+'\n'
            result_str+="pro_acc:"+str(acc_pro)+'\n'
            logger.info(result_str)
            result_str = '\n'
            logger.info("label evaluation done!")

    #if "relations" in iou_types:
    if cfg.PROTECTED.EVENT and not cfg.PROTECTED.MULTI_LABEL:
        logger.info("event evaluation start!")
        
        gt=[]
        pred=[]
        pred_indices=[]
        for image_id, prediction in enumerate(predictions):
            gt_event=dataset.get_event_target(image_id)
            gt.append(gt_event.detach().cpu().item())
            pred.append(int(prediction.get_field("pred_event_label").detach().cpu().item()))

            _, indices = prediction.get_field("el").detach().cpu()[0].topk(5)
            pred_indices.append(indices.tolist())

        # -------- Top-1 / Top-5 --------
        total_samples_event = 0
        top1_correct = 0
        top5_correct = 0
        for true, indices in zip(gt, pred_indices):
            total_samples_event += 1
            if true in indices:
                top5_correct += 1
                if true == indices[0]:
                    top1_correct += 1

        top1_accuracy = top1_correct / total_samples_event
        top5_accuracy = top5_correct / total_samples_event
        logger.info(f'Top1:{top1_accuracy}')
        logger.info(f'Top5:{top5_accuracy}')

        # -------- Accuracy & Recall --------
        from collections import defaultdict

        total_samples = len(gt)
        correct_predictions = 0
        class_correct = defaultdict(int)   # TP
        class_total = defaultdict(int)     # TP + FN (real samples per class)

        for true, predicted in zip(gt, pred):
            if true == predicted:
                correct_predictions += 1
                class_correct[true] += 1
            class_total[true] += 1

        # overall accuracy = micro recall in single-label tasks
        accuracy = correct_predictions / total_samples
        logger.info(f"Total Accuracy (Micro Recall): {accuracy:.5f}")

        # Per-class recall
        per_class_recalls = {}
        for class_label in sorted(class_total.keys()):
            recall = class_correct[class_label] / class_total[class_label]
            per_class_recalls[class_label] = recall
            logger.info(f"Class {class_label}: Recall={recall:.5f}")

        # Macro Recall = average of per-class recall
        macro_recall = sum(per_class_recalls.values()) / len(per_class_recalls)
        
        # Micro Recall = accuracy for single-label classification
        micro_recall = accuracy

        logger.info(f"Macro Recall: {macro_recall:.5f}")
        logger.info(f"Micro Recall: {micro_recall:.5f}")

        '''if cfg.PROTECTED.EVENT and not cfg.PROTECTED.MULTI_LABEL:
            logger.info("event evaluation start!")
            gt=[]
            pred=[]
            pred_indices=[]
            for image_id, prediction in enumerate(predictions):
                #gt_boxlist=dataset.get_protect_target(image_id,prediction.size)
                gt_event=dataset.get_event_target(image_id)
                #gt.append(gt_boxlist.get_field("event_label").detach().cpu().item())
                gt.append(gt_event.detach().cpu().item())
                pred.append(int(prediction.get_field("pred_event_label").detach().cpu().item()))

                _, indices = prediction.get_field("el").detach().cpu()[0].topk(5)
                pred_indices.append(indices.tolist())


            total_samples_event = 0
            top1_correct = 0
            top5_correct = 0
            for true, indices in zip(gt, pred_indices):
                total_samples_event += 1
                if true in indices:
                    top5_correct += 1
                    if true == indices[0]:
                        top1_correct += 1

            top1_accuracy = top1_correct / total_samples_event
            top5_accuracy = top5_correct / total_samples_event

            logger.info(f'Top1:{top1_accuracy}')
            logger.info(f'Top5:{top5_accuracy}')

            from collections import defaultdict

            total_samples = len(gt)
            correct_predictions = 0
            class_correct = defaultdict(int)
            class_total = defaultdict(int)

            for true, predicted in zip(gt, pred):
                if true == predicted:
                    correct_predictions += 1
                    class_correct[true] += 1
                class_total[true] += 1

            accuracy = correct_predictions / total_samples
            logger.info(f"Total Accuracy: {accuracy:.5f}")

            #for class_label in class_total.keys():
            for class_label in range(len(class_total.keys())):
                class_accuracy = class_correct[class_label] / class_total[class_label]
                #logger.info(f"{class_accuracy:.5f}")
                print(f"{class_accuracy:.5f}")'''
                
        if cfg.PROTECTED.EVENT and cfg.PROTECTED.MULTI_LABEL:
            logger.info("multi label event evaluation start!")
            gt=[]
            bpreds=[]
            preds=[]
            for image_id, prediction in enumerate(predictions):
                gt_event=dataset.get_event_target(image_id)
                gt.append(gt_event.detach().cpu().numpy())
                #pred.append(int(prediction.get_field("pred_event_label").detach().cpu().item()))
                pred = torch.sigmoid(prediction.get_field("el"))
                binary_pred = (pred >= 0.5).int()
                bpreds.append(binary_pred.view(-1).detach().cpu().numpy())
                preds.append(pred.view(-1).detach().cpu().numpy())
            gt=np.array(gt)
            bpreds=np.array(bpreds)
            preds=np.array(preds)
            
            label_accuracy = np.mean(bpreds == gt, axis=0)
            logger.info(f"label acc: {label_accuracy}")
            mean_label_accuracy = np.mean(label_accuracy)
            logger.info(f"avg label acc: {mean_label_accuracy}")
            
            correct_samples = np.all(bpreds == gt, axis=1)
            example_accuracy = np.mean(correct_samples)
            logger.info(f"example acc: {example_accuracy}")
            mAP = np.mean([average_precision_score(gt[:, i], preds[:, i]) for i in range(gt.shape[1])])
            logger.info(f"mAP: {mAP}")

        '''if cfg.PROTECTED.OBJ:

            logger.info("protect label evaluation start!")
            gt=[]
            pred=[]
            for image_id, prediction in enumerate(predictions):
                #for field in prediction.fields():
                    #print(field,prediction.get_field(field))
                gt_boxlist=dataset.get_protect_target(image_id,prediction.size)
                gt.append(gt_boxlist.get_field("protect_labels").detach().cpu().item())
                pred.append(int(prediction.get_field("pred_obj_label").detach().cpu().item()))

            from collections import defaultdict

            total_samples = len(gt)
            correct_predictions = 0
            class_correct = defaultdict(int)
            class_total = defaultdict(int)

            for true, predicted in zip(gt, pred):
                if true == predicted:
                    correct_predictions += 1
                    class_correct[true] += 1
                class_total[true] += 1

            accuracy = correct_predictions / total_samples
            logger.info(f"Total Accuracy: {accuracy:.5f}")

            for class_label in class_total.keys():
                class_accuracy = class_correct[class_label] / class_total[class_label]
                logger.info(f"Class {class_label} Accuracy: {class_accuracy:.5f}")
                #print(f"Class {class_label} Accuracy: {class_accuracy:.5f}")'''
        if cfg.PROTECTED.OBJ:

            logger.info("protect label evaluation start!")
            gt=[]
            pred=[]
            for image_id, prediction in enumerate(predictions):
                gt_boxlist = dataset.get_protect_target(image_id, prediction.size)
                gt.append(gt_boxlist.get_field("protect_labels").detach().cpu().item())
                pred.append(int(prediction.get_field("pred_obj_label").detach().cpu().item()))

            from collections import defaultdict

            total_samples = len(gt)
            correct_predictions = 0
            class_correct = defaultdict(int)  # TP per class
            class_total = defaultdict(int)    # TP + FN per class (real samples)

            # accumulate TP & class counts
            for true, predicted in zip(gt, pred):
                if true == predicted:
                    correct_predictions += 1
                    class_correct[true] += 1
                class_total[true] += 1

            # overall accuracy = micro recall (single-label classification)
            accuracy = correct_predictions / total_samples
            micro_recall = accuracy

            logger.info(f"Total Accuracy (Micro Recall): {accuracy:.5f}")

            # per-class recall = per-class accuracy in single-label setting
            per_class_recalls = []
            for class_label in sorted(class_total.keys()):
                recall = class_correct[class_label] / class_total[class_label]
                per_class_recalls.append(recall)
                logger.info(f"Class {class_label} Recall: {recall:.5f}")

            # macro recall = unweighted average of per-class recall
            macro_recall = sum(per_class_recalls) / len(per_class_recalls)
            logger.info(f"Macro Recall: {macro_recall:.5f}")
            logger.info(f"Micro Recall: {micro_recall:.5f}")

        if cfg.KGEVENT.PRIVACY_SCORE:
            epss=[]
            for image_id, prediction in enumerate(predictions):
                eps=prediction.get_field('eps')
                #eps=100*math.log(min(12,eps.item()+1),12)
                epss.append(eps)
            eps_avg=sum(epss)/len(epss)
            logger.info(f"Avg EPS: {eps_avg:.5f}")

    if cfg.PROTECTED.TEST:
        return predictions
    
    save_json=False
    file_scores={}
    if save_json:
        for prediction in predictions:
            file_scores[prediction.get_field('filename')]=prediction.get_field('eps').item()
        with open(os.path.join(output_folder, "event_explain_0720/llava_0.69.json"), 'w', encoding='utf-8') as f:
            json.dump(file_scores, f, ensure_ascii=False)
    
    
    
    return float(avg_metrics), result_dict_list_to_log, predictions


def save_output(output_folder, predictions, dataset):
    if output_folder:
        #print(predictions[0].fields())
        torch.save({'predictions': predictions},
                   os.path.join(output_folder, "event_explain_0720/llava_0.69.pytorch"))

        # with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # jupyter information
        '''visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
            ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)'''


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_scores').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        if evaluator.get("eval_pair_accuracy") is not None:
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return
    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    #print(local_container)
    # No Graph Constraint
    if evaluator.get("eval_nog_recall") is not None:
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    if evaluator.get("eval_pair_accuracy") is not None:
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    if evaluator.get("eval_mean_recall") is not None:
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)

    if evaluator.get("eval_ng_mean_recall") is not None:
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    if evaluator.get("eval_zeroshot_recall") is not None:
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # stage wise recall
    if evaluator.get("eval_stagewise_recall") is not None:
        evaluator['eval_stagewise_recall'] \
            .calculate_recall(mode, global_container,
                              gt_boxlist=groundtruth.convert('xyxy').to("cpu"),
                              gt_relations=groundtruth.get_field('relation_tuple').long().detach().cpu(),
                              pred_boxlist=prediction.convert('xyxy').to("cpu"),
                              pred_rel_pair_idx=prediction.get_field('rel_pair_idxs').long().detach().cpu(),
                              pred_rel_scores=prediction.get_field('pred_rel_scores').detach().cpu())
    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets
