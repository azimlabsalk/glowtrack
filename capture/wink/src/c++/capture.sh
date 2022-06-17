#!/bin/bash

DATA_DIR=$1
EVERY_NTH=$2
BUFFER_SIZE=$3
BATCH_SIZE=$4

# $ ./capture.sh some_dir 1 100 1 -> capture a batch every 1 frames, when not capturing buffer 100 frames, and capture 1 frame in each batch
# $ ./capture.sh some_dir 100 1 2 -> capture a every 100 frames, when not capturing buffer 1 frames, and capture 2 frames in each batch

./build/Grab_MultipleCameras $DATA_DIR $EVERY_NTH $BUFFER_SIZE $BATCH_SIZE
