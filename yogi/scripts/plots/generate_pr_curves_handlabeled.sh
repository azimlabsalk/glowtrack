#!/bin/bash

yogi plot pr-comparison-labelset output/fig4/pr-curves-cerebro-diverse-handlabeled.eps cerebro-clipgroup-test-hand-labeled right-paw cerebro-clipgroup-test-hand-labeled cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-train-1000-diverse cerebro-train-scale-0.8 --subsample-conf 1

yogi plot pr-comparison-labelset output/fig4/pr-curves-cerebro-diverse-handlabeled-fluor.eps cerebro-clipgroup-test-hand-labeled right-paw cerebro-all-behaviors-clean cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-train-1000-diverse cerebro-train-scale-0.8 --subsample-conf 1

