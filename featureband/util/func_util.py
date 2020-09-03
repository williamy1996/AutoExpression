import numpy as np

from sklearn.metrics import accuracy_score, f1_score


class Particle:

    def __init__(self, resource, feature_num, k):
        self.resource = resource
        self.k = k  # features to be selected
        if k is not None:
            self.feature_selected = np.random.choice(feature_num, self.k, replace=False).tolist()
            self.feature_selected.sort()
            self.position = indices2binary(self.feature_selected, feature_num)
        else:
            self.position = np.random.randint(0, 2, size=feature_num)
            self.feature_selected = binary2indices(self.position)

        self.performance = None

    """
    evaluate the fitness/metrics of a particle
    """
    def evaluate(self, x, y, clf, metrics="accuracy"):
        x_train, x_valid = x
        y_train, y_valid = y

        indices = self.feature_selected
        clf.fit(x_train[:, indices], y_train)
        y_pred = clf.predict(x_valid[:, indices])

        if metrics == "accuracy":
            self.performance = accuracy_score(y_valid, y_pred)
        if metrics == "f1":
            self.performance = f1_score(y_valid, y_pred)

    def mutation(self, global_best, sample_best):
        # d = len(self.position)
        # c1 = 0.5
        # c2 = 1.0 - c1
        # r1 = np.random.rand(d)
        # r2 = np.random.rand(d)
        #
        # # update the position
        # v = c1 * r1 * (global_best.position - self.position) + \
        #     c2 * r2 * (sample_best[self.resource].position - self.position)
        # s = 1 / (1 + np.exp(-v))
        # self.position = np.array([1 if si >= 0.5 else 0 for si in s])
        # self.feature_selected = binary2indices(self.position)

        d = len(self.position)
        m = int(d * 0.1 / 2)
        for _ in range(m):
            bit1, bit2 = np.random.choice(d, 2, replace=False)
            if self.position[bit1] + self.position[bit2] == 1:
                self.position[bit1] = 1 - self.position[bit1]
                self.position[bit2] = 1 - self.position[bit2]

    def __lt__(self, other):
        return self.performance < other.performance

    def __le__(self, other):
        return self.performance <= other.performance

    def __gt__(self, other):
        return self.performance > other.performance

    def __ge__(self, other):
        return self.performance >= other.performance

    def __str__(self):
        return "resource = " + str(self.resource) + ", performance = " + str(self.performance)


def crossover(chromosome1, chromosome2, m):
    assert len(chromosome1) == len(chromosome2)
    chromosome1 = list(chromosome1)
    chromosome2 = list(chromosome2)

    cutting_points = np.random.choice(len(chromosome1), m, replace=False).tolist()
    cutting_points.append(len(chromosome1))
    cutting_points.sort()
    result = []
    index = 0
    flag = 1

    for point in cutting_points:
        if flag > 0:
            result = result + chromosome1[index:point]
            index = point
            flag = -1
        elif flag < 0:
            result = result + chromosome2[index:point]
            index = point
            flag = 1

    return np.array(result, dtype=int)


def mutation_ga(chromosome, pm, k):
    n1 = sum(chromosome)
    n0 = len(chromosome) - n1
    p1 = pm
    p0 = pm * n1 / n0

    for i, g in enumerate(chromosome):
        r = np.random.rand()
        if g == 1 and r < p1:
            chromosome[i] = 0
        elif g == 0 and r < p0:
            chromosome[i] = 1

    if k is not None:
        # ensure the number of the selected features to be same
        selected_num = sum(chromosome)
        if selected_num == k:
            return
        if selected_num > k:
            one_index = [index for index in range(len(chromosome)) if chromosome[index] == 1]
            remove_index = np.random.choice(one_index, selected_num - k, replace=False)
            chromosome[remove_index] = 0
        if selected_num < k:
            zero_index = [index for index in range(len(chromosome)) if chromosome[index] == 0]
            remove_index = np.random.choice(zero_index, k - selected_num)
            chromosome[remove_index] = 1


def binary2indices(binary):
    indices = []
    for i, bit in enumerate(binary):
        if bit == 1:
            indices.append(i)
    return indices


def indices2binary(indices, feature_num):
    binary = [0] * feature_num
    for indice in indices:
        binary[indice] = 1
    return np.array(binary, dtype=int)


def local_mutation(particle):
    position = list(particle.position)
    d = len(position)
    m = int(d * 0.1 / 2)
    for _ in range(m):
        bit1, bit2 = np.random.choice(d, 2, replace=False)
        if position[bit1] + position[bit2] == 1:
            position[bit1] = 1 - position[bit1]
            position[bit2] = 1 - position[bit2]
    p = Particle(particle.resource, d, particle.k)
    p.position = np.array(position)
    p.feature_selected = binary2indices(p.position)
    return p