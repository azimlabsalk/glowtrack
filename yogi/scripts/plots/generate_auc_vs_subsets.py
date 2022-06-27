from yogi.evaluation import imageset_auc_table, subclipset_auc_table, auc_barchart
from yogi.db import session

# subset comparison table
df_subsets = subclipset_auc_table(session,
                          subclipset_names=[
                            'challenge-2-ayesha-pellet',
                            'challenge-2-elischa-pellet',
                            'challenge-2-bigpaw',
                            'challenge-2-nick-ropepull',
                            'challenge-2-elischa-water',
                            'challenge-2-elischa-treadmill',
                            'challenge-2-graziana-scratch',
                            'challenge-2-graziana-beam-leftward',
                            'challenge-2-graziana-beam-rightward-flipped',
                            'challenge-2-keewui-water-side-flipped',                            
                          ],
                          label_source_names=[
                            'cerebro-train-250-flipped',
                            'cerebro-train-500-flipped',
                            'cerebro-train-1000-flipped',
                            'cerebro-1000-diverse-flipped',
                            'cerebro-global-0.8-long-flipped',
                            'cerebro-global-0.8-long-flipped-autoscale-fast',
                          ],
                          gt_labelset_name='challenge-2-left-with-flipped',
                          landmarkset_name='left-paw',
#                          threshold_units='normalized', error_threshold=0.05,
                          threshold_units='digit-length', error_threshold=1.0,
                          use_occluded_gt=True)

df_subsets.to_csv('output/fig4/auc-challenge-2-subsets.csv')

from yogi.evaluation import auc_subclipset_barchart

auc_subclipset_barchart('output/fig4/auc-challenge-2-subsets.eps', 'output/fig4/auc-challenge-2-subsets.csv')
