import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GroupTesting:

    def __init__(self, p, clf, t):
        self.p = p  # prob of putting a feature in the test
        self.clf = clf
        self.t = t

        self.N = None
        self.score = None
        self.selected_features = None

    def fit(self, x, y):
        self.N = x.shape[1]  # N is the total number of input features
        if self.t is None:
            self.t = int(np.log2(self.N) + 1)
        self.score = np.zeros(self.t)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
        A = np.random.choice(2, size=(self.t, self.N), p=[0.5, 0.5])  # A is the test matrix
        for i in range(self.t):
            selected = A[i, :]
            x_train_selected = x_train[:, selected]
            x_valid_selected = x_valid[:, selected]
            self.clf.fit(x_train_selected, y_train)
            y_pred = self.clf.predict(x_valid_selected)
            accu = accuracy_score(y_valid, y_pred)
            self.score[i] = accu

        rho = np.zeros(self.N)
        for j in range(self.N):
            rho[j] = np.dot(self.score, A[:, j])

        self.selected_features = np.argsort(rho)[::-1]

    def transform(self, x, d):
        return x[:, self.selected_features[:d]]
