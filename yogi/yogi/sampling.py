import random


class Subsampler(object):

    valid_types = ['random', 'head', 'tail']

    def __init__(self, method, fraction):
        self.method = method
        self.fraction = fraction

    def sample(self, collection):
        k = len(collection) * self.fraction
        k = int(k)

        if self.method == 'random':
            subsample = random.sample(collection, k)
        elif self.method == 'head':
            subsample = collection[:k]
        elif self.method == 'tail':
            subsample = collection[-k:]

        return subsample


def random_pairs(max_value, number):
    pairs = [(n, n+1) for n in random.sample(range(max_value - 1), number)]
    return pairs


def non_overlapping_random_pairs(max_value, number):
    overlap = True
    while overlap:
        pairs = random_pairs(max_value, number)
        overlap = len(set([x for pair in pairs for x in pair])) != 2 * len(pairs)
    return pairs

