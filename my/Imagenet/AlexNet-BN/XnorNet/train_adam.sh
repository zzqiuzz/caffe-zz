set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -d "snapshot_adam" ]; then
	mkdir snapshot
fi
if [ ! -f "$WORK_SPACE/log_adam.log" ]; then
  touch "$WORK_SPACE/log_adam.log"
fi
LOG=$WORK_SPACE/log.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver_adam.prototxt  -gpu 6 2>&1 | tee $LOG

