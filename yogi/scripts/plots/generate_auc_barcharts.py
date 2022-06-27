from yogi.evaluation import auc_barchart

# panel 1
auc_barchart('output/fig4/auc-cerebro-1cam1light.eps', 'output/fig4/auc-cerebro-1cam1light.csv')


# panel 2
auc_barchart('output/fig4/auc-cerebro-diverse.eps', 'output/fig4/auc-cerebro-diverse.csv')


# challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-global-0.8-long-flipped cerebro-global-0.8-long-flipped-autoscale-fast

# panel 3
auc_barchart('output/fig4/auc-challenge-2-left.eps', 'output/fig4/auc-challenge-2-left.csv')


