#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="2,3"

exp_name="KG-event61-protect4"

if true; then
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/event_train_net.py \
    --config-file "configs/KG_event_WIDER.yaml" \
    EXPERIMENT_NAME "$exp_name" \
    SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 
fi

if false; then
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/event_train_net.py \
    --config-file "configs/event_WIDER.yaml" \
    EXPERIMENT_NAME "$exp_name" \
    SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 
fi
if false; then
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relBGNN_vg.yaml" \
    EXPERIMENT_NAME "$exp_name" \
    SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.VAL_PERIOD 1000 \
    SOLVER.CHECKPOINT_PERIOD 1000 
fi
