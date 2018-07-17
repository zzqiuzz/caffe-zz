set -e
TOOLS=./build/tools
LOG=my/BWN/VGG9/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/BWN/VGG9/solver.prototxt -gpu 0 2>&1 | tee $LOG	


