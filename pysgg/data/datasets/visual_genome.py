import json
import logging
import os
import random
from collections import defaultdict, OrderedDict, Counter
import pickle
import math
from torchvision import transforms

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from pysgg.config import cfg
from pysgg.structures.bounding_box import BoxList
from pysgg.structures.boxlist_ops import boxlist_iou, split_boxlist, cat_boxlist
from pysgg.utils.comm import get_rank, synchronize

from pysgg.data.datasets.bi_lvl_rsmp import resampling_dict_generation, apply_resampling

BOX_SCALE = 1024  # Scale at which we have the boxes

HEAD = []
BODY = []
TAIL = []

for i, cate in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
    if cate == 'h':
        HEAD.append(i)
    elif cate == 'b':
        BODY.append(i)
    elif cate == 't':
        TAIL.append(i)


# HEAD = [31, 20, 48, 30]
# BODY = [22, 29, 50, 8, 21, 1, 49, 40, 43, 23, 38, 41]
# TAIL = [6, 7, 46, 11, 33, 16, 9, 25, 47, 19, 35, 24, 5, 14, 13, 10, 44, 4, 12, 36, 32, 42, 26, 28, 45, 2, 17, 3, 18, 34,
#         37, 27, 39, 15]


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=1000, check_img_file=False,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        if cfg.DEBUG:
            num_im = 6000
            num_val_im = 600
        #
        #num_im = 20000
        num_val_im = 500

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        # self.remove_tail_classes = False
        self.drop_covered_obj=cfg.PECONFIG.DROP_COVERED_OBJ
        self.hideseeker=cfg.PECONFIG.HIDESEEKER

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("pysgg.dataset")
        self.logger = logger

        self.categories = {i: self.ind_to_classes[i]
                           for i in range(len(self.ind_to_classes))}

        self.protect_bbox = False
        if cfg.PECONFIG.USE_PROTECT_BOX:
            """precoompute boxes format:
                index by the image id, elem has "scores", "bbox", "cls", 3 fields
            """
            '''with open(os.path.join("datasets/vg/stanford_spilt", "detection_precompute_boxes_all.pkl"), 'rb') as f:
                self.pre_compute_bbox = pickle.load(f)
            self.logger.info("load pre-compute box length %d" %
                             (len(self.pre_compute_bbox.keys())))'''
            self.protect_bbox = True
            self.po_file=cfg.PECONFIG.PROTECT_DATA_FILE
            self.split_mask0,self.protect_obj=load_protect_obj(self.po_file)
                           
        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=False if not cfg.MODEL.RELATION_ON and split == "train" else True,
            filter_non_overlap=self.filter_non_overlap,
            split_mask0=None if not self.protect_bbox else self.split_mask0
        )

        self.filenames, self.img_info = load_image_filenames(
            img_dir, image_file, self.check_img_file)  # length equals to split_mask
        self.filenames = [self.filenames[i]
                          for i in np.where(self.split_mask)[0]]
        
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        '''if cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING and self.split == 'train':
            self.resampling_method = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD
            assert self.resampling_method in ['bilvl', 'lvis']

            self.global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
            self.drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE
            # creat repeat dict in main process, other process just wait and load
            if get_rank() == 0:
                repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)
                self.repeat_dict = repeat_dict
                with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
                    pickle.dump(self.repeat_dict, f)

            synchronize()
            self.repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)

            duplicate_idx_list = []
            for idx in range(len(self.filenames)):
                r_c = self.repeat_dict[idx]
                duplicate_idx_list.extend([idx for _ in range(r_c)])
            self.idx_list = duplicate_idx_list'''

        # if cfg.MODEL.ROI_RELATION_HEAD.REMOVE_TAIL_CLASSES and self.split == 'train':
        #     self.remove_tail_classes = True

    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.repeat_dict is not None:
            index = self.idx_list[index]

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        target = self.get_groundtruth(index, flip_img=False)
        # todo add pre-compute boxes
        protect_boxlist = None
        if self.protect_bbox:
            # index by image id
            '''pre_comp_result = self.pre_compute_bbox[int(
                self.img_info[index]['image_id'])]
            boxes_arr = torch.as_tensor(pre_comp_result['bbox']).reshape(-1, 4)
            pre_compute_boxlist = BoxList(boxes_arr, img.size, mode='xyxy')
            pre_compute_boxlist.add_field(
                "pred_scores", torch.as_tensor(pre_comp_result['scores']))
            pre_compute_boxlist.add_field(
                'pred_labels', torch.as_tensor(pre_comp_result['cls']))'''
            
            #pro_label=self.protect_obj[self.filenames[index]]['label']
            #pro_box=self.protect_obj[self.filenames[index]]['box']
            protect_boxlist=self.get_protect_target(index,img.size)

            img = np.array(img)
            x1, y1, x2, y2 = [int(num) for num in protect_boxlist.bbox[0].tolist()]
            img[y1:y2, x1:x2] = [255, 255, 255]
            img = Image.fromarray(img)
            img_input=img.copy()

        if self.transforms is not None:
            if protect_boxlist is not None:
                
                target1=BoxList(target.bbox, target.size, mode='xyxy')
                protect_boxlist1=BoxList(protect_boxlist.bbox, protect_boxlist.size, mode='xyxy')
                all_boxes = cat_boxlist([target1, protect_boxlist1])
                img, all_boxes = self.transforms(img, all_boxes)
                resized_boxes = split_boxlist(
                    all_boxes, (len(target), len(target) + len(protect_boxlist)))
                target1 = resized_boxes[0]
                protect_boxlist1 = resized_boxes[1]
                for field in target.fields():
                    target1.add_field(field, target.get_field(field))
                for field in protect_boxlist.fields():
                    protect_boxlist1.add_field(field, protect_boxlist.get_field(field))
                target=target1
                protect_boxlist=protect_boxlist1

                # cat the target and precompute boxes and transform them together
                '''targets_len = len(target)

                for field in target.fields():
                    protect_boxlist.add_field(field, target.get_field(field)[0].unsqueeze(0))
                target.add_field("protect_labels", torch.zeros((len(target))))
                
                all_boxes = cat_boxlist([target, protect_boxlist])
                img, all_boxes = self.transforms(img, all_boxes)
                resized_boxes = split_boxlist(
                    all_boxes, (targets_len, targets_len + len(protect_boxlist)))
                target = resized_boxes[0]
                target.remove_field("protect_labels")
                protect_boxlist = resized_boxes[1]
                for field in target.fields():
                    protect_boxlist.remove_field(field)'''

                is_protect_list=self.get_is_protect(target, protect_boxlist)
                target.add_field("is_protect",is_protect_list)
                #print(target.bbox,protect_boxlist.bbox)
            else:
                img, target = self.transforms(img, target)
        
        if protect_boxlist is None:
            return img, target, index
        else:
            to_tensor = transforms.ToTensor()
            protect_boxlist.add_field("index",index)
            if self.hideseeker:
                trans = transforms.Compose([
                    transforms.ToTensor(),
                ])
                protect_boxlist.add_field("image",trans(img_input))
        if self.drop_covered_obj:
            target=drop_obj(target,protect_boxlist)
        return img, target, protect_boxlist

    def get_is_protect(self,target,protect_boxlist):
        threshold=0.8
        match_quality_matrix = boxlist_iou(target, protect_boxlist)
        match_list=[item for sublist in match_quality_matrix.tolist() for item in sublist]
        match_arr=np.array(match_list)
        match_arr=np.where(match_arr > threshold, 1, 0)
        return torch.from_numpy(match_arr)

    def get_protect_target(self,index,img_size):
        trans_dir='/datapool'
        pro_label=self.protect_obj[trans_dir+self.filenames[index]]['label']
        pro_box=self.protect_obj[trans_dir+self.filenames[index]]['box']

        boxes_arr = np.array(pro_box) / BOX_SCALE * max(img_size[0],img_size[1])
        boxes_arr = torch.from_numpy(boxes_arr).reshape(-1, 4)  # guard against no boxes
        #boxes_arr = torch.as_tensor([pro_box]).reshape(-1, 4)
        protect_boxlist = BoxList(boxes_arr, img_size, mode='xyxy')
        protect_boxlist.add_field(
            'protect_labels', torch.as_tensor([pro_label]))
        
        return protect_boxlist

    def get_statistics(self):
        fg_matrix, bg_matrix, rel_counter_init = get_VG_statistics(self,
                                                 must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }

        rel_counter = Counter()

        for i in tqdm(self.idx_list):
            
            relation = self.relationships[i].copy()  # (num_rel, 3)
            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert self.split == 'train'
                old_size = relation.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in relation:
                    all_rel_sets[(o0, o1)].append(r)
                relation = [(k[0], k[1], np.random.choice(v))
                            for k, v in all_rel_sets.items()]
                relation = np.array(relation, dtype=np.int32)

            if self.repeat_dict is not None:
                relation, _ = apply_resampling(i, 
                                               relation,
                                               self.repeat_dict,
                                               self.drop_rate,)

            for i in relation[:, -1]:
                if i > 0:
                    rel_counter[i] += 1

        cate_num = []
        cate_num_init = []
        cate_set = []
        counter_name = []

        sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]
        lt_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        for cate_id in sorted_cate_list:
            if lt_part_dict[cate_id] == 'h':
                cate_set.append(0)
            if lt_part_dict[cate_id] == 'b':
                cate_set.append(1)
            if lt_part_dict[cate_id] == 't':
                cate_set.append(2)

            counter_name.append(self.ind_to_predicates[cate_id])  # list start from 0
            cate_num.append(rel_counter[cate_id])  # dict start from 1
            cate_num_init.append(rel_counter_init[cate_id])  # dict start from 1

        pallte = ['r', 'g', 'b']
        color = [pallte[idx] for idx in cate_set]


        fig, axs_c = plt.subplots(2, 1, figsize=(13, 10), tight_layout=True)
        fig.set_facecolor((1, 1, 1))

        axs_c[0].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
        axs_c[0].grid()
        plt.sca(axs_c[0])
        plt.xticks(rotation=-90, )

        axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
        axs_c[1].grid()
        axs_c[1].set_ylim(0, 50000)
        plt.sca(axs_c[1])
        plt.xticks(rotation=-90, )

        save_file = os.path.join(cfg.OUTPUT_DIR, f"rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)


        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        relation_non_masked = None
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index, 
                                                              relation,
                                                             self.repeat_dict,
                                                             self.drop_rate,)
        # add relation to target
        num_box = len(target)
        relation_map_non_masked = None
        if self.repeat_dict is not None:
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)


        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None  :
                        relation_map_non_masked[int(relation_non_masked[i, 0]), 
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None  :
                    relation_map_non_masked[int(relation_non_masked[i, 0]), 
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])


        target.add_field("relation", relation_map, is_triplet=True)
        if relation_map_non_masked is not None :
             target.add_field("relation_non_masked", relation_map_non_masked.long(), is_triplet=True)


        target = target.clip_to_image(remove_empty=False)
        target.add_field("relation_tuple", torch.LongTensor(
                relation))  # for evaluation
        return target

    def __len__(self):
        return len(self.idx_list)

def get_VG_statistics(train_data, must_overlap=True):
    """save the initial data distribution for the frequency bias model

    Args:
        train_data ([type]): the self
        must_overlap (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes,
                          num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    rel_counter = Counter()
    for ex_ind in tqdm(range(len(train_data.img_info))):
        gt_classes = train_data.gt_classes[ex_ind]
        gt_relations = train_data.relationships[ex_ind]
        gt_boxes = train_data.gt_boxes[ex_ind]

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
            rel_counter[gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix, rel_counter

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(
        np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, : 2],
                    boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:],
                    boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    print("correct img info")
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in tqdm(range(len(data)), total=len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(
        predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(
        attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file, check_img_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename) or not check_img_file:
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, split_mask0):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0
    
    split_mask=split_mask & split_mask0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[: num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[: num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]
    '''if split=='test':
        image_index = image_index[: 5]'''

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, : 2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0]
            == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end
                                 + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            # (num_rel, 3), representing sub, obj, and pred
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    #split_mask, boxes, gt_classes, gt_attributes, relationships=post_filter(split_mask, boxes, gt_classes, gt_attributes, relationships)
    return split_mask, boxes, gt_classes, gt_attributes, relationships

def load_protect_obj(po_file):
    with open(po_file, 'r') as f:
        data = json.load(f)
    return data['split_mask'], data['protect']

def post_filter(split_mask, boxes, gt_classes, gt_attributes, relationships):
    #sort_label=[107,92,131,72,9,85,88,75,77,115,65,44,46,149,78,91,90]
    sort_label=[107,92,72,9,85,88,75,77,115,44,46,91]

    i=0
    boxes1 = []
    gt_classes1 = []
    gt_attributes1 = []
    relationships1 = []
    for index in range(len(split_mask)):
        mask=split_mask[index]
        if mask==True:
            box=boxes[i]
            gt_class=gt_classes[i]
            gt_attribute=gt_attributes[i]
            relationship=relationships[i]
            flag=True
            for cls in gt_class:
                if cls in sort_label:
                    boxes1.append(box)
                    gt_classes1.append(gt_class)
                    gt_attributes1.append(gt_attribute)
                    relationships1.append(relationship)
                    flag=False
                    break
            if flag:
                split_mask[index]=False
            i=i+1
    return split_mask, boxes1, gt_classes1, gt_attributes1, relationships1

def calculate_intersection_area(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))

    area1=(y2-y1)*(x2-x1)
    area2=(y4-y3)*(x4-x3)

    cover1=intersection_area / area1
    cover2=intersection_area / area2

    return cover1,cover2

def drop_obj(target,protect_boxlist):
    th=0.2
    protect_bbox=protect_boxlist.bbox.detach().cpu().numpy()[0]
    is_covered=[]
    for bbox in target.bbox:
        bbox=bbox.detach().cpu().numpy()
        cover1,cover2=calculate_intersection_area(protect_bbox,bbox)
        if (cover2 > th and cover1 < 0.9) or (cover1>0.9 and cover2>th and cover2<0.9):
            is_covered.append(1)
        else:
            is_covered.append(0)
    #attr,rel_tuple,relation,labels,is_protect,bbox
    num=0
    trans_list=[]
    for i in is_covered:
        if i==0:
            trans_list.append(num)
            num+=1
        else:
            trans_list.append(-1)

    is_covered=torch.Tensor(is_covered)
    bbox_arr=target.bbox[is_covered==0]
    uncover_target = BoxList(bbox_arr, target.size, mode='xyxy')

    attributes=target.get_field("attributes")[is_covered==0]
    labels=target.get_field("labels")[is_covered==0]
    is_protect=target.get_field("is_protect")[is_covered==0]
    relation=target.get_field("relation")[is_covered==0]
    relation=relation[:,is_covered==0]
    ori_rel_tup=target.get_field("relation_tuple")

    relation_tuple=[]
    for i in range(len(ori_rel_tup)):
        k=ori_rel_tup[i].detach().cpu().numpy()
        if trans_list[int(k[0])]==-1 or trans_list[int(k[1])]==-1:
            continue
        k[0]=trans_list[int(k[0])]
        k[1]=trans_list[int(k[1])]
        relation_tuple.append(k)
    relation_tuple=torch.Tensor(relation_tuple).int()

    uncover_target.add_field("labels", labels)
    uncover_target.add_field("attributes", attributes)
    uncover_target.add_field("is_protect", is_protect)
    uncover_target.add_field("relation_tuple", relation_tuple)
    uncover_target.add_field("relation", relation)
    return uncover_target