set -e

TOOLS=./build/tools
LOG=my/Imagenet/AlexNet-BN/XnorNet_BWN/log.log
$TOOLS/caffe train \
    --solver=my/Imagenet/AlexNet-BN/XnorNet_BWN/solver.prototxt -gpu 0 2>&1 | tee $LOG

