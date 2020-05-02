#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES=2 python3 ETHTrain.py --regime regime_fp.json --epochs 100 --gray 1 --load-trainset "/home/nickyz/data/160x96HimaxMixedTrain_12_03_20AugCrop.pickle"  --save-model "Models/JabbaNet160x96.pt" 

