#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
FILE=$4

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
   $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --out /tmp/tmp$FILE.pkl --eval bbox
