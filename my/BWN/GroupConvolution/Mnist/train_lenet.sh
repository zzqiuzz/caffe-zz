#!/usr/bin/env sh
set -e
TOOLS=./build/tools
LOG=my/BWN/GroupConvolution/Mnist/snapshot/log.log
$TOOLS/caffe train --solver=my/BWN/GroupConvolution/Mnist/lenet_multistep_solver.prototxt --gpu 0 2>&1 | tee $LOG
