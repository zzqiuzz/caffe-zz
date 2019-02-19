set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -f "$WORK_SPACE/log_mean.log" ]; then
  touch "$WORK_SPACE/log_mean.log"
fi
LOG=$WORK_SPACE/log_mean.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver_mean.prototxt  -gpu 2 2>&1 | tee $LOG

