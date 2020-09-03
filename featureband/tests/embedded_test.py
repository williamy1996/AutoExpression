import numpy as np

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


DATASET = "coil20"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)

rf = RandomForestClassifier()
rf.fit(x, y)
sorted_features = np.argsort(rf.feature_importances_)[::-1]
rf.fit(x, y)

knn = load_clf("knn")
logistic = load_clf("logistic")

knn_perfs = np.zeros(10)
logistic_perfs = np.zeros(10)
skf = StratifiedKFold(n_splits=n_splits, random_state=42)
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for i, d in enumerate(np.arange(10, 101, 10)):
        x_train_selected = x_train[:, sorted_features[:d]]
        x_test_selected = x_test[:, sorted_features[:d]]

        knn.fit(x_train_selected, y_train)
        y_pred = knn.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)
        knn_perfs[i] += accu

        logistic.fit(x_train_selected, y_train)
        y_pred = logistic.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)
        logistic_perfs[i] += accu

print("splits:", n_splits)
knn_perfs /= n_splits
logistic_perfs /= n_splits
print("knn perfs:", knn_perfs.tolist())
print("logistic perfs:", logistic_perfs.tolist())
