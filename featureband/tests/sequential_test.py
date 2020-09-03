from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf
from featureband.sequential import sfs, sffs
from featureband.util.metrics_util import evaluate_cross_validation

from time import time

SEQUENTIAL_METHOD = "sffs"  # ["sfs", "sffs"]
DATASET = "madelon"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "decision_tree"]
k = 100

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

selected = []
if SEQUENTIAL_METHOD == "sfs":
    t0 = time()
    selected = sfs(x, y, k, clf)
    print("execution time:", time() - t0)

elif SEQUENTIAL_METHOD == "sffs":
    t0 = time()
    selected = sffs(x, y, k, clf)
    print("execution time:", time() - t0)

x_selected = x[:, selected]
print(k, SEQUENTIAL_METHOD, DATASET, FINAL_CLASSIFIER)
print("cross accuracy = ", evaluate_cross_validation(x_selected, y, clf, n_splits=10))
