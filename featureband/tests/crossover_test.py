import numpy as np
import os

from featureband.util.data_util import load_coil20, load_basehock
from featureband.feature_band import binary2indices
from featureband.genetic import mutation_ga

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from featureband.genetic import crossover


if __name__ == '__main__':
    data_path = "../dataset/MADELON"
    madelon = np.load(os.path.join(data_path, "madelon.npz"))
    x, y = madelon['x'], madelon['y']
    feature_num = x.shape[1]
    # clf = LogisticRegression()
    clf = KNeighborsClassifier(n_neighbors=3)
    skf = StratifiedKFold(n_splits=10)

    residuals = []
    residuals_random = []

    high_count = 0
    median_count = 0
    low_count = 0

    high_random = 0
    median_random = 0
    low_random = 0
    for t in range(100):
        print("t =", t, end=', ')
        parent1 = np.random.randint(0, 2, size=feature_num).tolist()
        parent2 = np.random.randint(0, 2, size=feature_num).tolist()

        offspring = crossover(parent1, parent2, 10)
        mutation_ga(offspring, 0.1)

        rand_feature = np.random.randint(0, 2, size=feature_num).tolist()

        accu1 = []
        accu2 = []
        accu_offspring = []
        accu_random = []
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(x_train[:, binary2indices(parent1)], y_train)
            y_pred1 = clf.predict(x_test[:, binary2indices(parent1)])
            accu1.append(accuracy_score(y_test, y_pred1))

            clf.fit(x_train[:, binary2indices(parent2)], y_train)
            y_pred2 = clf.predict(x_test[:, binary2indices(parent2)])
            accu2.append(accuracy_score(y_test, y_pred2))

            clf.fit(x_train[:, binary2indices(offspring)], y_train)
            y_pred3 = clf.predict(x_test[:, binary2indices(offspring)])
            accu_offspring.append(accuracy_score(y_test, y_pred3))

            clf.fit(x_train[:, binary2indices(rand_feature)], y_train)
            y_pred4 = clf.predict(x_test[:, binary2indices(rand_feature)])
            accu_random.append(accuracy_score(y_test, y_pred4))

        mean_accu1 = np.mean(accu1)
        mean_accu2 = np.mean(accu2)
        mean_accu_offspring = np.mean(accu_offspring)
        mean_accu_random = np.mean(accu_random)

        print("parent1:", mean_accu1, "parent2:", mean_accu2, "offspring:", mean_accu_offspring,
              "random:", mean_accu_random)

        residuals.append(mean_accu_offspring - (mean_accu1 + mean_accu2) / 2)
        residuals_random.append(mean_accu_random - (mean_accu1 + mean_accu2) / 2)

        if mean_accu_offspring > max(mean_accu1, mean_accu2):
            high_count += 1
        elif mean_accu_offspring < min(mean_accu1, mean_accu2):
            low_count += 1
        else:
            median_count += 1

        if mean_accu_random > max(mean_accu1, mean_accu2):
            high_random += 1
        elif mean_accu_random < min(mean_accu1, mean_accu2):
            low_random += 1
        else:
            median_random += 1

    np.save("../result/residuals_madelon_knn_100", residuals)
    np.save("../result/random_residuals_madelon_knn_100", residuals_random)

    print("high count =", high_count)
    print("median count =", median_count)
    print("low count =", low_count)

    print("--------------------------------")

    print("random high count =", high_random)
    print("random median count =", median_random)
    print("random low count =", low_random)
