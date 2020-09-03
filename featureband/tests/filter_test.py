import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf

FILTER_METHOD = "mi"  # ["ftest", "mi", "chi2"]
DATASET = "basehock"  # ["madelon", "basehock", "usps", "coil20", "gisette"]
FINAL_CLASSIFIER = "logistic"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

score_func = None
if FILTER_METHOD == "ftest":
    score_func = f_classif
if FILTER_METHOD == "mi":
    score_func = mutual_info_classif
if FILTER_METHOD == "chi2":
    score_func = chi2

skf = StratifiedKFold(n_splits=n_splits, random_state=42)

perfs = np.zeros(10)
fold_index = 0
for train_index, test_index in skf.split(x, y):
    print("fold:", fold_index + 1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    selector = SelectKBest(score_func=score_func)
    selector.fit(x_train, y_train)

    for i, k in enumerate(np.arange(10, 101, 10)):
        selector.k = k
        x_train_selected = selector.transform(x_train)
        x_test_selected = selector.transform(x_test)

        clf.fit(x_train_selected, y_train)
        y_pred = clf.predict(x_test_selected)

        accu = accuracy_score(y_test, y_pred)
        print("selected features:", k, "accu:", accu)
        perfs[i] += accu

    fold_index += 1

print("n_splits:", n_splits)
print(FILTER_METHOD, DATASET, FINAL_CLASSIFIER)
perfs /= 5
print(perfs.tolist())
