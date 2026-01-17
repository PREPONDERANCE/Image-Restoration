#!/usr/bin/env bash

CONFIG=$1

export CUDA_VISIBLE_DEVICES=0
python -m basicsr.train -opt $CONFIG --launcher pytorch
