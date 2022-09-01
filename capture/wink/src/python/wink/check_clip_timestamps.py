import sys
import pandas as pd
import numpy as np


path = sys.argv[1]

df = pd.read_csv(path, sep=' ', header=None)

ms = 5

frame1 = df[df[3] == 1][1].array
frame2 = df[df[3] == 2][1].array
frame_lags = frame2 - frame1

delta = ms * 1000000
errors = np.abs(frame_lags - delta)

n_bad = np.sum(errors > 50000)

print('n_bad / n_frames: {} / {}'.format(n_bad, len(df)))

