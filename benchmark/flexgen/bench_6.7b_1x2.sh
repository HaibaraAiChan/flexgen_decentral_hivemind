#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=2
N_CORES_PER_GPU=6

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=flexgen.dist_flex_opt

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl sm,self  \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  /home/cc/LLM/bin/python3.9 -m dist_flex_opt \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model facebook/opt-6.7b \
    --gpu-batch-size 48 \
    --percent 100 0 100 0 100 0 \
    --comm-device cpu \
    --cut-gen-len 5 \
    --path _DUMMY_
