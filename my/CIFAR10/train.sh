#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=my/CIFAR10/solver.prototxt --gpu 2 $@
 
