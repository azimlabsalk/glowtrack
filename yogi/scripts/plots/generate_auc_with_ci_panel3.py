from yogi.evaluation import imageset_auc_table, subclipset_auc_table
from yogi.db import session


# panel 1
#df_fig4_panel1 = imageset_auc_table(session, imageset_names=['cerebro-1-cam-1-light-test'],
#                          label_source_names=[
#                            'cerebro-train-250',
#                            'cerebro-train-500',
#                            'cerebro-train-1000',
#                            'cerebro-1000-diverse',
#                            'cerebro-global-0.8-long',
#                          ],
#                          gt_labelset_name='cerebro-1-cam-1-light-test',
#                          landmarkset_name='right-paw',
#                          threshold_units='normalized', error_threshold=0.05,
#                          use_occluded_gt=True)
#
#df_fig4_panel1.to_csv('output/fig4/auc-cerebro-1cam1light.csv')


# panel 2
#df_fig4_panel2 = imageset_auc_table(session, imageset_names=['cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small'],
#                          label_source_names=[
#                            'cerebro-train-250',
#                            'cerebro-train-500',
#                            'cerebro-train-1000',
#                            'cerebro-1000-diverse',
#                            'cerebro-global-0.8-long',
#                          ],
#                          gt_labelset_name='cerebro-clipgroup-train-minus-1cam1light-and-1000diverse-small',
#                          landmarkset_name='right-paw',
#                          threshold_units='normalized', error_threshold=0.05,
#                          use_occluded_gt=True)
#
#df_fig4_panel2.to_csv('output/fig4/auc-cerebro-diverse.csv')


# challenge-2-left-labeled left-paw challenge-2-labeled-dan cerebro-train-250-flipped cerebro-train-500-flipped cerebro-train-1000-flipped cerebro-1000-diverse-flipped cerebro-global-0.8-long-flipped cerebro-global-0.8-long-flipped-autoscale-fast

# panel 3
df_fig4_panel3 = imageset_auc_table(session, imageset_names=['challenge-2-left-labeled'],
                          label_source_names=[
                            'cerebro-train-250-flipped',
                            'cerebro-train-500-flipped',
                            'cerebro-train-1000-flipped',
                            'cerebro-1000-diverse-flipped',
                            'cerebro-global-0.8-long-flipped',
                            'cerebro-global-0.8-long-flipped-autoscale-fast',
                          ],
                          gt_labelset_name='challenge-2-labeled-dan',
                          landmarkset_name='left-paw',
                          threshold_units='normalized', error_threshold=0.05,
#                          threshold_units='digit-length', error_threshold=1.0,
                          use_occluded_gt=True)

df_fig4_panel3.to_csv('output/fig4/auc-challenge-2-left.csv')


