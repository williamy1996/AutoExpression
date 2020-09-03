import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./')
from featureband.feature_band import FeatureBand
from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import evaluate_cross_validation, load_clf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

DATASET = "rna"  # ["madelon", "basehock", "usps", "coil20"]
FINAL_CLASSIFIER = "logistic"  # ["knn", "logistic", "linear_svm"]
k = 300
n_splits = 5

r0 = 50
max_iter = 200
population_size = 10
n0 = 500

#x, y = load_dataset(DATASET)
x = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/Xtrain.csv'))
y = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/ytrain.csv',header=None))
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

skf = StratifiedKFold(n_splits=n_splits, random_state=42)
fold_index = 0
perfs = []
for train_index, test_index in skf.split(x, y):
    print("fold:", fold_index + 1)
    if fold_index != 0:
        continue
    #x_train, x_test = x[train_index], x[test_index]
    #y_train, y_test = y[train_index], y[test_index]
    x_train,y_train = x,y 
    fb = FeatureBand(r0=r0, n0=n0, clf=clf, max_iter=max_iter, k=k, population_size=population_size, local_search=True)
    times, iter_best, global_best = fb.fit(x_train, y_train, metrics="accuracy")

    x_train_selected = fb.transform(x_train)
    x_test_selected = fb.transform(x_test)
    print(x_train_selected.shape,'###############')
    clf.fit(x_train_selected, y_train)
    y_pred = clf.predict(x_test_selected)

    accu = accuracy_score(y_test, y_pred)
    print("accu:", accu)
    perfs.append(accu)

    fold_index += 1
    fname = DATASET + "_fb_" + str(k) + '_' + FINAL_CLASSIFIER + "_fold_" + str(fold_index)
    np.savez(os.path.join("./featureband/npdata", fname), times=times, iter_best=iter_best, global_best=global_best)
    break

print(DATASET, FINAL_CLASSIFIER, "k:", k)
print(perfs)
print(np.mean(perfs))
