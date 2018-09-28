set -e
TOOLS=./build/tools
LOG=my/Mnist/BWN/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/Mnist/BWN/lenet_multistep_solver.prototxt -gpu 0 2>&1 | tee $LOG	


