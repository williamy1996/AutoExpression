import numpy as np
import os

from featureband.feature_band import FeatureBand
from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import evaluate_cross_validation, load_clf

DATASET = "madelon"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
percentile = 10
r0 = 50
max_iter = 50
population_size = 5
n0 = 500

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
k = int(x.shape[1] * percentile / 100)
clf = load_clf(FINAL_CLASSIFIER)

fb = FeatureBand(r0=r0, n0=n0, clf=clf, max_iter=max_iter, k=k, population_size=population_size, local_search=False)
times, iter_best, global_best = fb.fit(x, y, metrics="accuracy")
x_selected = fb.transform(x)
print(x_selected.shape)

print(DATASET, FINAL_CLASSIFIER, "percentile:", percentile)
print("cross accuracy = ", evaluate_cross_validation(x_selected, y, clf, n_splits=10))

fname = DATASET + "_fbrandom_" + str(percentile) + '_' + FINAL_CLASSIFIER
np.savez(os.path.join("../log", fname), times=times, iter_best=iter_best, global_best=global_best)