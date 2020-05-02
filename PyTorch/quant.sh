#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES=3 python3 ETHQuantize2.py --regime regime.json --epochs 100 --gray 1 --load-trainset "/home/nickyz/data/160x96HimaxMixedTrain_12_03_20AugCrop.pickle" --load-model "Models/HannaNetFrancesco_290420_q8_f8.pt" --quantize --save-model "HannaNetQ.pt"

