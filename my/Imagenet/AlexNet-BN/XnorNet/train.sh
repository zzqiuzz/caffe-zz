set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -d "snapshot" ]; then
	mkdir snapshot
fi
if [ ! -f "$WORK_SPACE/log.log" ]; then
  touch "$WORK_SPACE/log.log"
fi
LOG=$WORK_SPACE/log.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver.prototxt --snapshot=snapshot/solver_iter_110000.solverstate -gpu 0 2>&1 | tee $LOG

