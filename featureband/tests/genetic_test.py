import numpy as np
import os

from featureband.genetic import GA
from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf, evaluate_cross_validation

# Set parameters
DATASET = "madelon"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
k = 100
max_iter = 10000

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

ga = GA(population_size=20, k=k, max_iter=max_iter, q=0.25)
times, iter_best, global_best = ga.fit(x, y, clf)
x_selected = ga.transform(x)
print(x_selected.shape)

print(DATASET, FINAL_CLASSIFIER, "k:", k)
print("cross accuracy = ", evaluate_cross_validation(x_selected, y, clf, n_splits=10))

fname = DATASET + '_ga_' + str(k) + '_' + FINAL_CLASSIFIER
np.savez(os.path.join("../npdata", fname), times=times, iter_best=iter_best, global_best=global_best)
