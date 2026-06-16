#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="0,1"

if false; then #explain
archive_dir="/workspace/huangyunyi/code/PySGG/configs/KG_event_mydata.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_protected.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net_explain.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2024-11-16_20_mydata-original-kg_resampling/model_0015000.pth" \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('MYEVENT_test', )" \
KGEVENT.EXPLAIN "GNNExplainer"
fi

#"/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2024-12-13_12_mydata-original-kg_resampling/model_0015000.pth" \

if false; then #VISPR
archive_dir="/workspace/huangyunyi/code/PySGG/configs/KG_multi_VISPR.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_mydata.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_protected.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2025-01-13_15_lgoi-pixel9_resampling/model_0005000.pth" \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('VISPR_test', )"
fi

if false; then #LGOI
archive_dir="/workspace/huangyunyi/code/PySGG/configs/KG_obj_LGOI.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_mydata.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_protected.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2025-01-06_13_lgoi-blur31_resampling/model_0005000.pth"  \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('LGOI_test', )"
fi

#"/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2025-01-06_13_lgoi-blur31_resampling/model_0005000.pth" 

if false; then #mydata
archive_dir="/workspace/huangyunyi/code/PySGG/configs/KG_event_mydata.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_mydata.yaml" #"/workspace/huangyunyi/code/PySGG/configs/KG_event_protected.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2025-01-06_13_lgoi-blur31_resampling/model_0005000.pth" \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('MYEVENT_test', )"
fi

#"/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2024-05-08_22_mydata_resampling/model_0007000.pth"
#"/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2024-11-16_20_mydata-original-kg_resampling/model_0015000.pth"

if true; then #wider
archive_dir="/workspace/huangyunyi/code/PySGG/configs/KG_event_WIDER.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2026-05-13_20_mydata-wo-ontology_resampling/model_0001000.pth" \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('EVENT61_test', )"
fi

if false; then
archive_dir="/workspace/huangyunyi/code/PySGG/configs/e2e_relBGNN_vg.yaml"
python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num \
tools/relation_test_net.py \
--config-file "$archive_dir" \
TEST.IMS_PER_BATCH $[$gpu_num] \
MODEL.WEIGHT  "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2023-11-11_16_BGNN-union_mask_resampling/model_0002000.pth" \
MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
DATASETS.TEST "('VG_stanford_filtered_with_attribute_test', )"
fi

