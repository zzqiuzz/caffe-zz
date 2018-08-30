set -e
touch log.log
TOOLS=./build/tools
LOG=my/Mnist/BWN_INQ_Relax/log.log
$TOOLS/caffe train \
    --solver=my/Mnist/BWN_INQ_Relax/lenet_multistep_solver.prototxt  --weights=my/Mnist/BWN_INQ_Relax/LeNet_b_99.caffemodel -gpu 0 2>&1 | tee $LOG	


