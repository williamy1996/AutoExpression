import numpy as np
import os

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = "./dataset/MADELON"
madelon = np.load(os.path.join(data_path, "madelon.npz"))
x, y = madelon['x'], madelon['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)


lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC(kernel="linear", max_iter=100)

clf = svm

sfm = SelectFromModel(estimator=clf, threshold=0.0008)
print(clf.coef_.__class__)
sfm.fit(x_train, y_train)
x_train_selected = sfm.transform(x_train)
x_test_selected = sfm.transform(x_test)
print(x_train_selected.shape, x_test_selected.shape)
clf.fit(x_train_selected, y_train)
y_pred = clf.predict(x_test_selected)

print("accuracy = ", accuracy_score(y_test, y_pred))
