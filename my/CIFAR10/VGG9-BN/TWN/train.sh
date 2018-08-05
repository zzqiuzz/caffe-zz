set -e
TOOLS=./build/tools
LOG=my/CIFAR10/VGG9-BN/TWN/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/CIFAR10/VGG9-BN/TWN/solver.prototxt -gpu 0 2>&1 | tee $LOG

