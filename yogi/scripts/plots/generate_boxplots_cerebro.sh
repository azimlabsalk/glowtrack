#!/bin/bash

# FIG 4

# yogi plot error-boxplot-labelset output/fig4/error-boxplot-cerebro-diverse.eps cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small right-paw cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-global-0.8-long

# panel 1
yogi plot error-boxplot-labelset output/fig4/error-boxplot-cerebro-1cam1light.eps cerebro-1-cam-1-light-test right-paw cerebro-1-cam-1-light-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-global-0.8-long --whis 0 --logplot 1

# panel 2
yogi plot error-boxplot-labelset output/fig4/error-boxplot-cerebro-diverse.eps cerebro-clipgroup-test-0.05 right-paw cerebro-clipgroup-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-train-1000-diverse cerebro-train-scale-0.8 --whis 0 --logplot 1

# panel 3
# yogi plot error-boxplot-labelset output/fig4/error-boxplot-challenge-2-left.eps challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-global-0.8-long-flipped cerebro-global-0.8-long-flipped-autoscale-fast 

# yogi plot error-boxplot-labelset output/fig4/error-boxplot-challenge-2-left-everything.eps challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-smoothed-widescale-flipped cerebro-smoothed-widescale-flipped-autoscale-conf cerebro-smoothed-flipped-autoscale-conf cerebro-global-0.8-long-flipped-autoscale-fast cerebro-global-0.8-long-flipped-autoscale 

