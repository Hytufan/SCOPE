#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="1"

#export NCCL_DEBUG=INFO
#export NCCL_SHM_DISABLE=1

exp_name="mydata-wo-ontology"

if true; then
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/pa_event_train_net.py \
    --config-file "/workspace/huangyunyi/code/PySGG/configs/KG_event_WIDER.yaml" \
    EXPERIMENT_NAME "$exp_name" \
    SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 6000 
fi

#MODEL.PRETRAINED_DETECTOR_CKPT "/workspace/huangyunyi/code/PySGG/checkpoints/sgdet-BGNNPredictor/2024-12-24_22_lgoi_resampling/model_0015000.pth" \

if false; then
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/pa_text_train_net.py \
    --config-file "/workspace/huangyunyi/code/PySGG/configs/KG_text_CoNLL.yaml" \
    EXPERIMENT_NAME "$exp_name" \
    SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 
fi