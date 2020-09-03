import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from time import time
from time import clock

from featureband.util.data_util import load_dataset
from featureband.util.metrics_util import evaluate_cross_validation

x, y = load_dataset("news20")
print(x.shape, y.shape)

# clf = LinearSVC()
clf = KNeighborsClassifier(n_neighbors=3)
# clf = LogisticRegression()

samples = np.random.choice(100, 5, replace=False)
x_selected = x[:, samples]

t0 = time()
c0 = clock()
perf = evaluate_cross_validation(x, y, clf, n_splits=10)
print("t:", time() - t0, "c:", clock() - c0)

print("perf:", perf)
