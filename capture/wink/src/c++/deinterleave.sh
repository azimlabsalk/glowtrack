#!/bin/bash

DATA_DIR=$1

for F in $DATA_DIR/*/*/interlaced.mp4;
do
  echo $F;
  DIR=`dirname $F`;
  mkdir $DIR/uv;
  mkdir $DIR/visible;
  python ../python/wink/deinterleave.py $F $DIR/visible/video.mp4 $DIR/uv/video.mp4;
done
