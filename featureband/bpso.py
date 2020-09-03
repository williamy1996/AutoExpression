from __future__ import print_function, division, absolute_import

from sklearn.model_selection import train_test_split


class BPSO:
    def __init__(self, swarm_size, max_iter, clf):
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.clf = clf

        self.data_num = None
        self.feature_num = None
        self.featrue_selected = None

    def fit(self, x, y, metrics):

        pass

    def transform(self):
        pass
