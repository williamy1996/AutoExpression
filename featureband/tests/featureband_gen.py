import numpy as np
import pandas as pd
import os
import sys
import pandas as pd
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
max_iter = 50
population_size = 10
n0 = 500

#x, y = load_dataset(DATASET)
x = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/Xtrain.csv'))
y = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/ytrain.csv',header=None))
x_test = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/Xtest.csv'))
y_test = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/ytest.csv',header=None))
print(x.shape, y.shape)
clf = load_clf(FINAL_CLASSIFIER)

x_train,y_train = x,y[:,0]
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
pd.DataFrame(x_train_selected).to_csv('./medicaldata/tpotfssRNASeq/Xtrain_fb300.csv',index=False)
pd.DataFrame(x_test_selected).to_csv('./medicaldata/tpotfssRNASeq/Xtest_fb300.csv',index=False)

print(DATASET, FINAL_CLASSIFIER, "k:", k)
print(perfs)
print(np.mean(perfs))
