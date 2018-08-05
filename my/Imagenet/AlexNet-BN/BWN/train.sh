set -e
TOOLS=./build/tools
LOG=my/BWN/AlexNet-BN/snapshot/log.log
$TOOLS/caffe train \
    --solver=solver.prototxt -gpu 0 2>&1 | tee $LOG

