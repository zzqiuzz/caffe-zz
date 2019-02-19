set -e

CAFFE_ROOT=$HOME/caffe-zz
TOOLS=$CAFFE_ROOT/build/tools
WORK_SPACE=$(cd `dirname $0`; pwd)
if [ ! -f "$WORK_SPACE/log1.log" ]; then
  touch "$WORK_SPACE/log1.log"
fi
if [ ! -d "$WORK_SPACE/snapshot1" ]; then
  mkdir snapshot1
fi
LOG=$WORK_SPACE/log1.log
$TOOLS/caffe train \
    --solver=$WORK_SPACE/solver1.prototxt  -gpu 0 2>&1 | tee $LOG

