#!/bin/bash

INPUT_DIR=$1
OUTPUT_VIDEO=$2

ffmpeg -y -f image2 -framerate 24 -pattern_type glob -i "$INPUT_DIR"'/*.png' -c:v libx264 -preset veryslow -qp 18 -pix_fmt yuv420p $OUTPUT_VIDEO
