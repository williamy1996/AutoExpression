import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.linear_model import LogisticRegression

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf

DATASET = "coil20"  # ["madelon", "basehock", "usps", "coil20", "gisette"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)
# clf = LogisticRegression()

skf = StratifiedKFold(n_splits=n_splits, random_state=42)

fold_index = 0
perfs = []
for train_index, test_index in skf.split(x, y):
    print("fold:", fold_index + 1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    perfs.append(accuracy_score(y_test, y_pred))

print("n_splits:", n_splits)
print(DATASET, FINAL_CLASSIFIER)
print(perfs)
print(np.mean(perfs))
