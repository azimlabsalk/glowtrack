import sys

import numpy as np


path = sys.argv[1]

with open(path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

data = [[int(s) for s in line.strip().split(' ')] for line in lines]

ms1 = 15
ms2 = 5
ncams = 8

for cam in range(ncams):

    times = [row[1] for row in data if row[0] == cam]
    times = np.array(times)

    frame_lags = times[1:] - times[:-1]
    delta1 = ms1 * 1000000
    delta2 = ms2 * 1000000
    diffs1 = np.abs(frame_lags - delta1)
    diffs2 = np.abs(frame_lags - delta2)
    n_bad = np.sum(diffs1[::2] > 50000) + np.sum(diffs2[1::2] > 50000)

#    for diff in diffs:
#        print(diff)

    print('n_bad / n_frames: {} / {}'.format(n_bad, len(times)))

