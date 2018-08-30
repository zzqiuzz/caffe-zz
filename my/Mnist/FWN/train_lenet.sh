#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=my/Mnist/FWN/lenet_multistep_solver.prototxt --gpu all  $@
