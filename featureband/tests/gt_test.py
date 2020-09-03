import numpy as np

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf
from featureband.group_testing import GroupTesting
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


DATASET = "basehock"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)
perfs = np.zeros(10)
skf = StratifiedKFold(n_splits=n_splits, random_state=42)
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    gt = GroupTesting(p=0.5, clf=clf, t=50)
    gt.fit(x_train, y_train)
    for i, d in enumerate(np.arange(10, 101, 10)):
        x_train_selected = gt.transform(x_train, d)
        x_test_selected = gt.transform(x_test, d)

        clf.fit(x_train_selected, y_train)
        y_pred = clf.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)

        perfs[i] += accu

print("n_splits:", n_splits)
print("group testing", DATASET)
print("clf", FINAL_CLASSIFIER)
perfs /= n_splits
print("perfs:\n", perfs.tolist())
