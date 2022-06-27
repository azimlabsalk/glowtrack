#!/bin/bash

# FIG 6

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test.eps hand-test-labeled human-right-hand hand-test-labeled-dan right-hand-speckle-scale-0.25-4m right-hand-speckle-scale-0.25-4m-autoscale-conf right-hand-speckle-scale-0.25-bg-4m right-hand-speckle-scale-0.25-bg-4m-autoscale-conf --subsample-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-all.eps hand-test-labeled human-right-hand hand-test-labeled-dan right-hand-speckle right-hand-speckle-rotate right-hand-speckle-rotate-bg right-hand-speckle-scale-0.25 right-hand-speckle-scale-0.25-bg right-hand-speckle-scale-0.25-long right-hand-speckle-scale-0.25-bg-2m right-hand-speckle-scale-0.25-bg-2m-autoscale-conf right-hand-speckle-scale-0.25-4m right-hand-speckle-scale-0.25-4m-autoscale-conf right-hand-speckle-scale-0.25-bg-4m right-hand-speckle-scale-0.25-bg-4m-autoscale-conf --subsample-conf 1

yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-ntemplates.eps hand-test-labeled human-right-hand hand-test-labeled-dan speckle-5-features-median-1-templates-nn speckle-5-features-median-5-templates-nn speckle-5-features-median-10-templates-nn --subsample-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-aug.eps hand-test-labeled human-right-hand hand-test-labeled-dan right-hand-speckle right-hand-speckle-rotate right-hand-speckle-rotate-bg --subsample-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-samples.eps hand-test-labeled human-right-hand hand-test-labeled-dan right-hand-speckle-scale-0.25 right-hand-speckle-scale-0.25-long right-hand-speckle-scale-0.25-4m --subsample-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-bg.eps hand-test-labeled human-right-hand hand-test-labeled-dan right-hand-speckle-scale-0.25 right-hand-speckle-scale-0.25-bg right-hand-speckle-scale-0.25-long right-hand-speckle-scale-0.25-bg-2m right-hand-speckle-scale-0.25-4m right-hand-speckle-scale-0.25-bg-4m --subsample-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-labels-2.eps hand-speckle-labeled human-right-hand hand-speckle-labeled-dan-2 speckle-5-features-median speckle-5-features-median-1-templates speckle-5-features-median-2-templates speckle-4-features-median-3-templates speckle-5-features-median-4-templates speckle-5-features-median-5-templates speckle-5-features-median-10-templates --no-conf 1

# yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-sift.eps hand-test-labeled human-right-digit-1 hand-test-labeled-dan right-hand-speckle-scale-0.25-bg-2m-autoscale-conf stabilized-sift-1-top-50-autoscale-conf-nearest unstabilized-sift-1-top-50-autoscale-conf-nearest
yogi plot pr-comparison-labelset output/fig6/pr-curves-hand-test-sift.eps hand-test-labeled human-right-digit-1 hand-test-labeled-dan right-hand-speckle-scale-0.25-bg-2m-autoscale-conf stabilized-sift-1-top-50-scale-0.5-autoscale-conf-nearest unstabilized-sift-1-top-50-autoscale-conf-nearest stabilized-sift-1-top-50-scale-0.5-autoscale-min-error stabilized-sift-1-top-50-scale-0.35-autoscale-nearest
