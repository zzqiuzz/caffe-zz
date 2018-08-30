set -e
touch log.log
TOOLS=./build/tools
LOG=my/CIFAR10/VGG9-BN/BWN_INQ_Relax/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/CIFAR10/VGG9-BN/BWN_INQ_Relax/solver.prototxt --weights=my/CIFAR10/VGG9-BN/BWN_INQ_Relax/VGG9_BN_0.9118.caffemodel -gpu 0 2>&1 | tee $LOG	


