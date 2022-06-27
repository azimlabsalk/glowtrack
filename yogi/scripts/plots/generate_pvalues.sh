#!/bin/bash


yogi plot error-pvalues-labelset error-cdf-cerebro-1cam1light.eps cerebro-1-cam-1-light-test right-paw cerebro-1-cam-1-light-test cerebro-train-250 cerebro-train-500 cerebro-train-1000

yogi plot error-pvalues-labelset error-cdf-cerebro-diverse.eps cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small right-paw cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small cerebro-train-250 cerebro-train-500 cerebro-train-1000 cerebro-1000-diverse

yogi plot error-pvalues-labelset error-cdf-mnle.eps mnle-labeled left-paw mnle-labeled cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-global-0.8-long-flipped

yogi plot error-pvalues-labelset error-cdf-mnle-ntraining.eps mnle-labeled left-paw mnle-labeled cerebro-smoothed-flipped cerebro-smoothed-train-0.1-flipped cerebro-1000-diverse-flipped



# yogi plot pr-comparison pr-curves-scale-mnle.eps mnle-labeled left-paw mnle-labeled cerebro-scale-0.5-flipped cerebro-scale-0.6-flipped cerebro-scale-0.7-flipped cerebro-scale-0.8-flipped cerebro-scale-0.9-flipped cerebro-global-0.8-long-flipped cerebro-scale-1.1-flipped cerebro-scale-1.2-flipped cerebro-scale-1.3-flipped cerebro-scale-1.4-flipped--subsample-conf 1

# yogi plot pr-comparison pr-curves-scale-mnle.eps mnle-dev-rope left-paw mnle-labeled cerebro-scale-0.9-flipped cerebro-global-0.8-long-flipped cerebro-scale-1.1-flipped --subsample-conf 1

# yogi plot pr-comparison pr-curves-autoscaling-mnle.eps mnle-dev left-paw mnle-labeled cerebro-global-0.8-long-flipped cerebro-global-0.8-long-flipped-autoscale cerebro-global-0.8-long-flipped-autoscale-fast cerebro-global-0.6-widescale-flipped cerebro-global-0.6-widescale-flipped-autoscale cerebro-global-0.6-widescale-flipped-autoscale-conf --subsample-conf 1

# yogi plot error-histogram error_histogram_cerebro_smoothed.eps mnle-dev left-paw mnle-labeled cerebro-smoothed-flipped

# yogi plot pr-comparison pr-curves-mnle-bigpaw.eps mnle-dev left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1
# yogi plot pr-comparison pr-curves-mnle-rope-bigpaw.eps mnle-dev-rope left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1
# yogi plot pr-comparison pr-curves-mnle-reach-bigpaw.eps mnle-dev-reach left-paw mnle-labeled cerebro-global-0.8-long-flipped-autoscale-fast bigpaw-autoscale-conf --subsample-conf 1

# yogi plot pr-comparison pr-curves-challenge-dev-easy-autoscale.eps challenge-dev-easy-no-mnle right-paw challenge-labeled cerebro-global-0.8-long-autoscale-conf bigpaw-flipped-autoscale-conf --subsample-conf 1

