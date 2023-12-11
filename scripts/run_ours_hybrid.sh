#!/bin/bash

MY_IPADDR=127.0.1.1
N_GPUS=4

# nbatch 
NB=64

PYTHON_EXEC=python3

# Hybrid Parallelism
PYTHON_SCRIPT=flexgen.__dist_flex_opt
BATCH_SIZE=$(($NB/$N_GPUS))

mpirun \
  --allow-run-as-root \
  --map-by ppr:$N_GPUS:node \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7776 \
    --use-mpi \
	--model facebook/opt-6.7b \
    --gpu-batch-size $BATCH_SIZE \
    --comm-device gpu
