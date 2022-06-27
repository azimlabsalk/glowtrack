#!/bin/bash

GPU=$1

echo "GPU = $GPU"

for MODEL in cerebro-1-cam-1-light-aug cerebro-4-cam-1-light-aug cerebro-aug cerebro-all-behaviors-aug cerebro-paired-mono-aug cerebro-paired-color-aug;
do
  echo $MODEL;
  CUDA_VISIBLE_DEVICES=$GPU yogi label from-model $MODEL published-videos-random-0.01;
  CUDA_VISIBLE_DEVICES=$GPU yogi label from-model $MODEL cerebro-paired-color-random-0.01;
  CUDA_VISIBLE_DEVICES=$GPU yogi label from-model $MODEL challenge-set-dev;
done

