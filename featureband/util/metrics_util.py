import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


def evaluate_cross_validation(x, y, clf, n_splits, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    accus = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accus.append(accuracy_score(y_test, y_pred))

    return np.mean(accus)


def evaluate_holdout(x, y, clf, test_size, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return accuracy_score(y_test, y_pred)


def load_clf(clf):
    if clf == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    if clf == "logistic":
        return LogisticRegression()
    if clf == "decision_tree":
        return DecisionTreeClassifier()
    if clf == "linear_svm":
        return LinearSVC()
