import numpy as np

from sklearn.model_selection import train_test_split
from time import clock

from featureband.util.func_util import Particle
from featureband.util.func_util import crossover, mutation_ga
from featureband.util.func_util import binary2indices


class GA:

    def __init__(self, population_size, k, max_iter, q):
        self.population_size = population_size
        self.k = k
        self.max_iter = max_iter
        self.q = q

        self.population = None
        self.data_num = None
        self.feature_num = None
        self.feature_selected = None

    def select_chromosome(self):
        prob = [0]
        for i in range(1, len(self.population) + 1):
            prob.append(self.q * (1 - self.q) ** (i - 1))
        prob = np.cumsum(prob)
        r = np.random.uniform(0, prob[-1])

        for i in range(1, len(prob)):
            if prob[i - 1] < r < prob[i]:
                return i - 1

    def fit(self, x, y, clf):
        assert len(x) == len(y)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
        self.data_num, self.feature_num = x_train.shape
        self.population = [Particle(None, self.feature_num, self.k) for _ in range(self.population_size)]

        # initialization
        for p in self.population:
            p.evaluate(x=[x_train, x_valid], y=[y_train, y_valid], clf=clf)
        self.population.sort(reverse=True)

        global_best = []
        iter_best = []
        times = []
        t0 = clock()
        for t in range(self.max_iter):
            # selection
            p1 = self.population[self.select_chromosome()]
            p2 = self.population[self.select_chromosome()]
            offspring = Particle(None, self.feature_num, self.k)

            # crossover
            ch = crossover(p1.position, p2.position, 10)

            # mutation
            mutation_ga(ch, 0.1, self.k)
            offspring.position = ch
            offspring.feature_selected = binary2indices(ch)

            # replace
            offspring.evaluate(x=[x_train, x_valid], y=[y_train, y_valid], clf=clf)
            self.population.append(offspring)
            self.population.sort(reverse=True)
            self.population = self.population[:self.population_size]

            times.append(clock() - t0)
            iter_best.append(offspring.performance)
            global_best.append(self.population[0].performance)

            print("t:", t, "iter_best:", offspring.performance, "global_best:", self.population[0].performance)

        self.feature_selected = binary2indices(self.population[0].position)
        return times, iter_best, global_best

    def transform(self, x):
        return x[:, self.feature_selected]


if __name__ == '__main__':
    # test crossover
    # ch1 = list(np.random.randint(0, 2, size=20))
    # ch2 = list(np.random.randint(0, 2, size=20))

    # print("before crossover:")
    # print("ch1:", ch1)
    # print("ch2:", ch2)
    # ch3 = crossover(ch1, ch2)
    # print(ch3)

    # test mutation_ga
    ch = np.random.randint(0, 2, size=30)
    print("before mutation, ch:", ch)
    mutation_ga(ch, 0.1, 10)
    print("after  mutation, ch:", ch)
