import numpy as np

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import load_clf
from featureband.CCM.core import ccm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


DATASET = "usps"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "linear_svm"]
n_splits = 5

x, y = load_dataset(DATASET)

n_samples = np.random.choice(x.shape[0], 3000, replace=False)
x = x[n_samples, :]
y = y[n_samples]


print(x.shape, y.shape)
knn = load_clf("knn")
logistic = load_clf("logistic")


knn_perfs = np.zeros(10)
logistic_perfs = np.zeros(10)
skf = StratifiedKFold(n_splits=n_splits, random_state=42)
fold_index = 0
fout = open("usps_data.txt", 'w')
for train_index, test_index in skf.split(x, y):
    print("fold:", fold_index + 1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for i, k in enumerate(np.arange(10, 101, 10)):
        rank = ccm.ccm(x_train, y_train, k, "ordinal", 0.001, iterations=10, verbose=True)
        idx = np.argsort(rank).tolist()[:k]
        x_train_selected = x_train[:, idx]
        x_test_selected = x_test[:, idx]

        knn.fit(x_train_selected, y_train)
        y_pred = knn.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)

        fout.write("selected features: " + str(k) + "knn_accu: " + str(accu) + '\n')
        knn_perfs[i] += accu

        logistic.fit(x_train_selected, y_train)
        y_pred = logistic.predict(x_test_selected)
        accu = accuracy_score(y_test, y_pred)

        fout.write("selected features: " + str(k) + "logistic_accu: " + str(accu) + '\n')
        fout.flush()
        logistic_perfs[i] += accu

    fold_index += 1

fout.write("n_splits: " + str(n_splits) + "\n")
fout.write("ccm" + str(DATASET) + "knn, logistic\n")
knn_perfs /= n_splits
logistic_perfs /= n_splits
fout.write("knn perfs:\n" + str(knn_perfs) + "\n")
fout.write("logistic perfs:\n" + str(logistic_perfs) + "\n")
fout.flush()
