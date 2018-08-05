set -e
TOOLS=./build/tools
LOG=my/CIFAR10/VGG9-BN/BWN/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/CIFAR10/VGG9-BN/BWN/solver.prototxt -gpu 0 2>&1 | tee $LOG	


