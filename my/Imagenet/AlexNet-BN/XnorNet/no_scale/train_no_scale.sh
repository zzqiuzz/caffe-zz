set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -d "snapshot_no_scale" ]; then
	mkdir snapshot_no_scale
fi
if [ ! -f "$WORK_SPACE/log_no_scale.log" ]; then
  touch "$WORK_SPACE/log_no_scale.log"
fi
LOG=$WORK_SPACE/log_no_scale.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver_no_scale.prototxt   -gpu 3 2>&1 | tee $LOG

