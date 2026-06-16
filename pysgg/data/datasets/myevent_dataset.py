import logging
import numpy as np
import os
import torch
import json
from pysgg.structures.bounding_box import BoxList
from torchvision import transforms
from pysgg.config import cfg
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm
from pysgg.data.datasets.bi_lvl_rsmp import resampling_dict_generation, apply_resampling
import matplotlib.pyplot as plt
import h5py
from pysgg.structures.boxlist_ops import boxlist_iou, split_boxlist, cat_boxlist
from PIL import Image, ImageFilter

def list_files_in_directory(directory_path):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.basename(root)+'/'+file)
    return file_list


BOX_SCALE = 1024  # Scale at which we have the boxes

class MYEVENTDataset(torch.utils.data.Dataset):
    def __init__(self, split, img_dir, filename_file_path, dict_file, roidb_file, transforms=None,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False):
        self.image_dir = img_dir
        #self.image_dir = '/workspace/huangyunyi/data/MyEventDataset/prs_0115/random_obj/0.30/' #'/workspace/huangyunyi/data/MyEventDataset/prs_0115/all_face_white/'
        #self.image_dir = ''
        self.filename_file= filename_file_path+f'{split}.txt'
        self.filename_file='/workspace/huangyunyi/code/PySGG/checkpoints/inference/MYEVENT_test/event_explain_0115/score_0.5.txt'
        #self.filename_file='/workspace/huangyunyi/data/NIO/images/filelist/14.txt'
        self.transforms = transforms
        self.hideseeker=cfg.PECONFIG.HIDESEEKER
        self.split=split
        self.kgevent=cfg.PECONFIG.KG_EVENT
        self.obj_and_event= True #cfg.PROTECTED.OBJ # check object?
        self.test_mode= False #cfg.PROTECTED.TEST  # dir or file?
        self.has_gt = True # event GT?
        self.has_root_dir=False

        if self.obj_and_event:
            self.protect_info_filename=cfg.PECONFIG.PROTECT_DATA_FILE
            #self.protect_info_filename='/workspace/huangyunyi/res/info.json'
            with open(self.protect_info_filename, 'r') as file:
                self.protect_info = json.load(file)

            '''with open('/workspace/huangyunyi/data/MyEventDataset/protected_info/explain_score_v1/explain_1.json', 'r') as file:
                self.protect_info = json.load(file)'''

        if self.test_mode:
            #self.filenames=list(self.protect_info.keys())
            #target_list=[0]*len(self.filenames)

            target_list=[]
            self.root_dir=self.image_dir #'/workspace/huangyunyi/data/MyEventDataset/explain_v1/explain_blur_0.99/'
            self.filenames=list_files_in_directory(self.root_dir)
            for file in self.filenames:
                if self.has_gt:
                    target_list.append(int(file.split('/')[0]))
                else:
                    target_list.append(0)
            for i in range(len(self.filenames)):
                self.filenames[i]=self.root_dir+self.filenames[i]
        else:
            self.filenames=[]
            target_list=[]
            with open(self.filename_file, 'r') as file:
                for line in file:
                    line=line.strip()
                    filename=line.split(' ')[0]
                    target=line.split(' ')[1]
                    target=int(target)

                    if not os.path.exists(self.image_dir+filename):
                        continue
                    if not self.obj_and_event:
                        self.filenames.append(filename)
                        target_list.append(target)
                    else:
                        if filename in self.protect_info.keys():
                            self.filenames.append(filename)
                            target_list.append(target)

        self.img_info=[]
        '''with open(self.target_file, 'r') as json_file:
            target_data = json.load(json_file)
        for filename in self.filenames:
            self.img_info.append(target_data[filename])'''
        for i in range(len(self.filenames)):
            if self.has_root_dir:
                image = Image.open(self.filenames[i])
            else:
                image = Image.open(os.path.join(self.image_dir+self.filenames[i]))
            width, height = image.size
            info={}
            info['img_path']=self.filenames[i]
            info['scene_id']=target_list[i]
            info['height']=height
            info['width']=width
            self.img_info.append(info)

        self.idx_list = list(range(len(self.filenames)))

        # VG stat
        self.flip_aug = flip_aug
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.roidb_file = roidb_file
        self.repeat_dict = None

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        
        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, -1, num_val_im=500,
            filter_empty_rels=False if not cfg.MODEL.RELATION_ON and split == "train" else True,
            filter_non_overlap=self.filter_non_overlap,
            split_mask0=None
        )


    def __getitem__(self, idx):
        if not self.has_root_dir:
            img = Image.open(os.path.join(self.image_dir+self.filenames[idx])).convert("RGB")
        else:
            img = Image.open(self.filenames[idx]).convert("RGB")
        #target_event=torch.as_tensor([self.img_info[idx]["scene_id"]])
        if self.obj_and_event:
            protect_boxlist=self.get_protect_target(idx,img.size)
        else:
            protect_boxlist=BoxList([[0,0,0,0]], img.size, mode='xyxy')
            protect_boxlist.add_field('protect_labels', torch.as_tensor([0]))
            target_event=torch.as_tensor([self.img_info[idx]["scene_id"]])
            protect_boxlist.add_field("event_label",target_event)
            protect_boxlist.add_field("is_protect",torch.as_tensor([1]))

        if self.obj_and_event:
            img=img
            img = np.array(img)
            x1, y1, x2, y2 = [int(num) for num in protect_boxlist.bbox[0].tolist()]
            img[y1:y2, x1:x2] = [255,255,255]#[255, 255, 255]
            img = Image.fromarray(img)
            '''bboxes=self.protect_info_extra[self.filenames[idx]]["bboxes"]
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = [int(num) for num in bboxes[i]]
                img[y1:y2, x1:x2] = [255, 255, 255]
            img = Image.fromarray(img)'''
            #bbox=[int(num) for num in protect_boxlist.bbox[0].tolist()]
            #img=apply_gaussian_blur(img,bbox,k=7)
            #img=apply_mosaic(img,bbox,mosaic_size=25)
            #img=masking_out(img,protect_boxlist,[255,255,255])
        #print(f'1 {protect_boxlist},{img.size}')

        img_input=img.copy()
        #protect_boxlist=None
        if self.transforms is not None:
            if protect_boxlist is not None:
                #print(protect_boxlist,protect_boxlist.bbox,self.filenames[idx])
                img, protect_boxlist = self.transforms(img, protect_boxlist)
                #print(protect_boxlist,protect_boxlist.bbox)

        #print(f'2 {protect_boxlist},{img.shape}')

        target=None
        '''target=protect_boxlist.copy()
        is_protect_list=self.get_is_protect(target, protect_boxlist)
        target.add_field("is_protect",is_protect_list)
        target.add_field("event_label",target_event)'''
        #protect_boxlist.add_field("event_label",target_event)
        #protect_boxlist.add_field("is_protect",torch.as_tensor([1]))
        
        to_tensor = transforms.ToTensor()
        protect_boxlist.add_field("index",idx)
        protect_boxlist.add_field("file_name",self.filenames[idx])
        if self.hideseeker or self.kgevent:
            trans = transforms.Compose([
                transforms.ToTensor(),
            ])
            protect_boxlist.add_field("image",trans(img_input))

        return img, target, protect_boxlist
    
    def get_img(self,idx):
        if not self.has_root_dir:
            img = Image.open(os.path.join(self.image_dir+self.filenames[idx])).convert("RGB")
        else:
            img = Image.open(self.filenames[idx]).convert("RGB")
        if self.obj_and_event:
            protect_boxlist=self.get_protect_target(idx,img.size)
        else:
            protect_boxlist=BoxList([[0,0,0,0]], img.size, mode='xyxy')
            protect_boxlist.add_field('protect_labels', torch.as_tensor([91]))
            target_event=torch.as_tensor([self.img_info[idx]["scene_id"]])
            protect_boxlist.add_field("event_label",target_event)
            protect_boxlist.add_field("is_protect",torch.as_tensor([1]))

        if self.obj_and_event:
            '''img = np.array(img)
            x1, y1, x2, y2 = [int(num) for num in protect_boxlist.bbox[0].tolist()]
            img[y1:y2, x1:x2] = [255, 255, 255]
            img = Image.fromarray(img)'''
            #bbox=[int(num) for num in protect_boxlist.bbox[0].tolist()]
            #img=apply_gaussian_blur(img,bbox)
            #img=apply_mosaic(img,bbox)
            img=img

        return img

    def get_is_protect(self,target,protect_boxlist):
        threshold=0.8
        match_quality_matrix = boxlist_iou(target, protect_boxlist)
        match_list=[item for sublist in match_quality_matrix.tolist() for item in sublist]
        match_arr=np.array(match_list)
        match_arr=np.where(match_arr > threshold, 1, 0)
        return torch.from_numpy(match_arr)
    
    def get_protect_target(self,index,img_size):

        #pro_label=self.img_info[index]['protect_label']
        #pro_box=self.img_info[index]['protect_bbox']
        
        filename=self.filenames[index]
        if self.test_mode:
            filename=self.filenames[index].replace(self.root_dir,'')

        pro_labels= self.protect_info[filename]["labels"]
        pro_bboxes = self.protect_info[filename]["bboxes"]

        boxes_arr = np.array(pro_bboxes) #/ BOX_SCALE * max(img_size[0],img_size[1])
        boxes_arr = torch.from_numpy(boxes_arr).reshape(-1, 4)  # guard against no boxes
        #boxes_arr = torch.as_tensor(pro_bboxes).reshape(-1, 4)
        protect_boxlist = BoxList(boxes_arr, img_size, mode='xyxy')
        #protect_boxlist=protect_boxlist.convert('xyxy')

        protect_boxlist.add_field(
            'protect_labels', torch.as_tensor(pro_labels))

        protect_boxlist.add_field("is_protect",torch.as_tensor([1]))
        target_event=torch.as_tensor([self.img_info[index]["scene_id"]])
        protect_boxlist.add_field("event_label",target_event)
        
        return protect_boxlist

    def get_event_target(self,index):
        return torch.as_tensor([self.img_info[index]["scene_id"]])

    def __len__(self):
        return len(self.filenames)

    def get_img_info(self,index):
        return self.img_info[index]
    
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
    
    if split_mask0!=None:
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

def apply_gaussian_blur(image, bounding_box,k):
    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Extract coordinates of bounding box
    x_min, y_min, x_max, y_max = bounding_box

    # Ensure coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_array.shape[1], x_max)
    y_max = min(img_array.shape[0], y_max)

    # Crop the image to the bounding box
    cropped_img = Image.fromarray(img_array[y_min:y_max, x_min:x_max])

    # Apply Gaussian blur
    blurred_img = cropped_img.filter(ImageFilter.GaussianBlur(radius=k))

    # Replace the bounding box region with the blurred content
    img_array[y_min:y_max, x_min:x_max] = np.array(blurred_img)

    # Convert the NumPy array back to PIL image
    result_image = Image.fromarray(img_array)

    return result_image

def masking_out(img,protect_boxlist,color=[255, 255, 255]):
    img = np.array(img)
    x1, y1, x2, y2 = [int(num) for num in protect_boxlist.bbox[0].tolist()]
    img[y1:y2, x1:x2] = color
    img = Image.fromarray(img)
    return img

def apply_mosaic(image, bounding_box, mosaic_size=25):
    # 获取图像的宽度和高度
    img_width, img_height = image.size

    # 检查边界框是否越界
    left, top, right, bottom = bounding_box
    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    right = max(0, min(right, img_width - 1))
    bottom = max(0, min(bottom, img_height - 1))

    # 获取边界框内的内容
    region = image.crop((left, top, right, bottom))

    # 将边界框内的内容进行马赛克处理
    region = region.resize((region.width // mosaic_size + 1, region.height // mosaic_size + 1), resample=Image.NEAREST)
    region = region.resize((region.width * mosaic_size, region.height * mosaic_size), resample=Image.NEAREST)

    # 将处理后的内容粘贴回原图像
    image.paste(region, (left, top))

    return image
