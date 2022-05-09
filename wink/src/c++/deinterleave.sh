#!/bin/bash

DATA_DIR=$1

for F in $DATA_DIR/*/*/interlaced.mp4;
do
  echo $F;
  DIR=`dirname $F`;
  mkdir $DIR/uv;
  mkdir $DIR/visible;
  python ../python/wink/deinterlace.py $F $DIR/uv/video.mp4 $DIR/visible/video.mp4;
done
