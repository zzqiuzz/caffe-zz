set -e
touch log.log
TOOLS=./build/tools
LOG=my/Imagenet/AlexNet_INQ_Relax/log.log
$TOOLS/caffe train \
    --solver=my/Imagenet/AlexNet_INQ_Relax/solver.prototxt \
    --weights=my/Imagenet/AlexNet_INQ_Relax/bvlc_alexnet_bn.caffemodel \
    --gpu 0 2>&1 | tee $LOG

