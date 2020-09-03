import numpy as np

from skfeature.function.information_theoretical_based import CMIM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf, evaluate_cross_validation

DATASET = "basehock"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "logistic"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)

print(x[:, 15].std())
print(x[:, 15])


print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

perfs = np.zeros(10)

skf = StratifiedKFold(n_splits=n_splits, random_state=42)
fold_index = 0
for train_index, test_index in skf.split(x, y):
    print("fold:", fold_index + 1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for i, k in enumerate(np.arange(10, 101, 10)):
        idx, _, _ = CMIM.cmim(x_train, y_train, n_selected_features=k)
        x_train_selected = x_train[:, idx[0:k]]
        x_test_selected = x_test[:, idx[0:k]]

        clf.fit(x_train_selected, y_train)
        y_pred = clf.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)

        print("selected features:", k, "accu:", accu)
        perfs[i] += accu

    fold_index += 1

print("n_splits:", n_splits)
print("cmim", DATASET, FINAL_CLASSIFIER)
# perfs /= n_splits
print(perfs)
