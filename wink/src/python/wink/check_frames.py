import sys

import numpy as np


path = sys.argv[1]

with open(path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

ms = 5

times = [int(line) for line in lines]
times = np.array(times)

frame_lags = times[1:] - times[:-1]
delta_t = ms * 1000000
diffs = np.abs(frame_lags - delta_t)
n_bad = np.sum(diffs > 50000)

for diff in diffs:
    print(diff)

print('n_bad: {}'.format(n_bad))

