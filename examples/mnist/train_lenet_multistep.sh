#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_multistep_solver.prototxt --gpu all $@
