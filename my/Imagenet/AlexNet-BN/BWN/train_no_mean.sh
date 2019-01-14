set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -f "$WORK_SPACE/log_no_mean.log" ]; then
  touch "$WORK_SPACE/log_no_mean.log"
fi
if [ ! -d "snapshot_no_mean" ]; then
	mkdir snapshot_no_mean
fi
LOG=$WORK_SPACE/log_no_mean.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver_no_mean.prototxt  --snapshot=snapshot_no_mean/solver_no_mean_iter_60000.solverstate -gpu 2 2>&1 | tee $LOG

