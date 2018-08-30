set -e
touch snapshot/log.log
TOOLS=./build/tools
LOG=my/Mnist/BWN/snapshot/log.log
$TOOLS/caffe train \
    --solver=my/Mnist/BWN/lenet_multistpe_solver.prototxt -gpu 0 2>&1 | tee $LOG	


