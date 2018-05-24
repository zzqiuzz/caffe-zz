#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/binary/lenet_multistep_solver.prototxt --gpu all  $@
