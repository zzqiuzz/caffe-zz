set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -f "$WORK_SPACE/log.log" ]; then
  touch "$WORK_SPACE/log.log"
fi
LOG=$WORK_SPACE/log.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver.prototxt  -gpu 5 2>&1 | tee $LOG

