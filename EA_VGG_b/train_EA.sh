#!/usr/bin/env sh
set -e 
TOOLS=./build/tools
LOG=EA_VGG_b/snapshot/log_b.log
$TOOLS/caffe train --solver=EA_VGG_b/Mini_VGG_gap_solver_b.prototxt --gpu 0 2>&1 | tee $LOG

