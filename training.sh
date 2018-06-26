#!/usr/bin/env bash
SOLVER=$1
CLASS=$2
LOG=/home/freedom/caffe/tianchifusai/InceptionV3/20180529/log/${CLASS}.log
TOOLS=/home/freedom/caffe/build/tools
GPU0=$3


echo "starting to train"

sudo $TOOLS/caffe train -solver $SOLVER -weights /home/freedom/caffe/tianchifusai/InceptionV3/20180522/inception-v4.caffemodel  -gpu $GPU0 2>&1 | tee ${LOG} 

