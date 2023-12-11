#!/bin/bash

MY_IPADDR=127.0.1.1
N_GPUS=4

# nbatch 
NB=32

PYTHON_EXEC=python3

# Pipeline Parallelism
PYTHON_SCRIPT=flexgen.dist_flex_opt
BATCH_SIZE=$(($NB/$N_GPUS))

# Data Parallelism
#PYTHON_SCRIPT=flexgen._dist_flex_opt
#BATCH_SIZE=$NB

# Hybrid Parallelism
#PYTHON_SCRIPT=flexgen.__dist_flex_opt
#BATCH_SIZE=$(($NB/$N_GPUS))

#pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

#python3 -m flexgen.flex_opt --model facebook/opt-6.7b 
#python3 -m flexgen.flex_opt --model facebook/opt-6.7b --debug-mode breakdown

#set -x

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

#--comm-device cpu 
#--model facebook/opt-6.7b \
#--model facebook/opt-6.7b \
