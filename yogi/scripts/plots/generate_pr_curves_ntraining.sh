#!/bin/bash

# FIG 4

# panel 2
# yogi plot pr-comparison-labelset output/fig4/pr-curves-cerebro-ntraining-new.eps cerebro-clipgroup-test-0.05 right-paw cerebro-clipgroup-test cerebro-clipgroup-train-5000 cerebro-clipgroup-train-20000 cerebro-train-scale-0.8 --subsample-conf 1

# panel 3

# yogi plot pr-comparison-labelset output/fig4/pr-curves-challenge-ntraining-new.eps challenge-2-left-with-flipped-labeled left-paw challenge-2-left-with-flipped cerebro-clipgroup-train-5000-flipped cerebro-clipgroup-train-20000-flipped cerebro-train-scale-0.8-flipped --subsample-conf 1

yogi plot pr-comparison-labelset output/fig4/pr-curves-challenge-ntraining-autoscale.eps challenge-2-left-with-flipped-labeled left-paw challenge-2-left-with-flipped cerebro-1000-diverse-flipped-autoscale-conf cerebro-clipgroup-train-5000-flipped-autoscale-conf cerebro-clipgroup-train-20000-flipped-autoscale-conf cerebro-train-scale-0.8-flipped-autoscale-conf --subsample-conf 1

#yogi plot pr-comparison-labelset output/fig4/pr-curves-challenge-2-left-everything.eps challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-smoothed-widescale-flipped cerebro-smoothed-widescale-flipped-autoscale-conf cerebro-smoothed-flipped-autoscale-conf cerebro-global-0.8-long-flipped-autoscale-fast cerebro-global-0.8-long-flipped-autoscale --subsample-conf 1

# perf vs. n training
# yogi plot pr-comparison-labelset pr-curves-mnle-ntraining.eps mnle-labeled left-paw mnle-labeled cerebro-smoothed-flipped cerebro-smoothed-train-0.1-flipped cerebro-smoothed-train-0.025-flipped cerebro-train-1000-flipped --subsample-conf 1
#yogi plot pr-comparison-labelset output/fig4/pr-curves-cerebro-diverse-ntraining.eps cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small right-paw cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small cerebro-smoothed cerebro-smoothed-train-0.1 cerebro-smoothed-train-0.025 cerebro-train-1000-diverse --subsample-conf 1
#yogi plot pr-comparison-labelset output/fig4/pr-curves-challenge-2-left-ntraining.eps challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-smoothed-flipped cerebro-smoothed-train-0.1-flipped cerebro-smoothed-train-0.025-flipped cerebro-train-1000-diverse-flipped --subsample-conf 1

# plot 2 alternates
# yogi plot pr-comparison-labelset pr-curves-cerebro-3C1L.eps cerebro-3-cam-1-light-test right-paw cerebro-4-cam-1-light-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-clipgroup-v1 --subsample-conf 1
# yogi plot pr-comparison-labelset pr-curves-cerebro-1C8L.eps cerebro-1-cam-8-light-test right-paw cerebro-1-cam-9-light-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-clipgroup-v1 --subsample-conf 1
# yogi plot pr-comparison-labelset pr-curves-cerebro-4C9L.eps cerebro-4-cam-9-light-test-0.05 right-paw cerebro-clipgroup-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-clipgroup-v1 --subsample-conf 1
# yogi plot pr-comparison-labelset pr-curves-cerebro-8C9L.eps cerebro-clipgroup-test-0.05 right-paw cerebro-clipgroup-test cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse cerebro-clipgroup-v1 --subsample-conf 1

# panel 2 bigpaw version
# yogi plot pr-comparison-labelset pr-curves-cerebro-diverse-bigpaw.eps bigpaw23-random-0.01 left-paw bigpaw23-random-0.01 cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-global-0.8-long-flipped --subsample-conf 1

# panel 2 bigpaw various models
# yogi plot pr-comparison-labelset pr-curves-cerebro-widescale-bigpaw.eps bigpaw23-random-0.01 left-paw bigpaw23-random-0.01 cerebro-global-0.6-widescale-flipped cerebro-smoothed-widescale-flipped cerebro-smoothed-widescale-flipped-autoscale-conf cerebro-smoothed-flipped cerebro-smoothed-flipped-autoscale-conf --subsample-conf 1



# perf vs. n training (bigpaw)
# yogi plot pr-comparison-labelset pr-curves-cerebro-ntraining-bigpaw.eps bigpaw23-random-0.01 left-paw bigpaw23-random-0.01 cerebro-smoothed-flipped cerebro-smoothed-train-0.1-flipped cerebro-smoothed-train-0.025-flipped cerebro-train-1000-flipped --subsample-conf 1
 
# extras
# yogi plot pr-comparison-labelset pr-curves-mnle-extras.eps mnle-dev left-paw mnle-labeled cerebro-smoothed-flipped cerebro-no-augm-flipped cerebro-smoothed-widescale-flipped cerebro-smoothed-widescale-flipped-autoscale-conf --subsample-conf 1

# yogi plot pr-comparison pr-curves-scale-mnle.eps mnle-labeled left-paw mnle-labeled cerebro-scale-0.5-flipped cerebro-scale-0.6-flipped cerebro-scale-0.7-flipped cerebro-scale-0.8-flipped cerebro-scale-0.9-flipped cerebro-global-0.8-long-flipped cerebro-scale-1.1-flipped cerebro-scale-1.2-flipped cerebro-scale-1.3-flipped cerebro-scale-1.4-flipped--subsample-conf 1

# yogi plot pr-comparison pr-curves-scale-mnle.eps mnle-dev-rope left-paw mnle-labeled cerebro-scale-0.9-flipped cerebro-global-0.8-long-flipped cerebro-scale-1.1-flipped --subsample-conf 1

# yogi plot pr-comparison pr-curves-autoscaling-mnle.eps mnle-dev left-paw mnle-labeled cerebro-global-0.8-long-flipped cerebro-global-0.8-long-flipped-autoscale cerebro-global-0.8-long-flipped-autoscale-fast cerebro-global-0.6-widescale-flipped cerebro-global-0.6-widescale-flipped-autoscale cerebro-global-0.6-widescale-flipped-autoscale-conf --subsample-conf 1

# yogi plot error-histogram error_histogram_cerebro_smoothed.eps mnle-dev left-paw mnle-labeled cerebro-smoothed-flipped

# yogi plot pr-comparison pr-curves-mnle-bigpaw.eps mnle-dev left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1
# yogi plot pr-comparison pr-curves-mnle-rope-bigpaw.eps mnle-dev-rope left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1
# yogi plot pr-comparison pr-curves-mnle-reach-bigpaw.eps mnle-dev-reach left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1

# yogi plot pr-comparison pr-curves-challenge-dev-easy-autoscale.eps challenge-dev-easy-no-mnle right-paw challenge-labeled cerebro-global-0.8-long-autoscale-conf bigpaw-flipped-autoscale-conf --subsample-conf 1

