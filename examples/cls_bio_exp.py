import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from solnml.bioestimator import Classifier

X_train = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/Xtrain.csv'))
X_test = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/Xtest.csv'))
y_train = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/ytrain.csv',header=None))
y_test = np.array(pd.read_csv('./medicaldata/tpotfssRNASeq/ytest.csv',header=None))
clf = Classifier(time_limit=3600,fb_max_iter=50)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
fa = clf.feature_analysis()
fa.to_csv('feature_analysis.csv')
print(accuracy_score(y_test, pred))
