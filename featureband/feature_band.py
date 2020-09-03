from __future__ import print_function, division, absolute_import

import numpy as np
import math

from sklearn.model_selection import train_test_split
from time import clock

from featureband.util.func_util import Particle
from featureband.util.func_util import crossover, mutation_ga
from featureband.util.func_util import binary2indices, indices2binary


class FeatureBand:

    def __init__(self, r0, n0, clf, max_iter, population_size, local_search, k=None, eta=3.0):
        self.r0 = r0  # initial size
        self.n0 = n0
        self.clf = clf  # the classifier to evaluate the performance
        self.max_iter = max_iter
        self.k = k  # the num of features to be selected
        self.eta = eta
        self.population_size = population_size
        self.local_search = local_search

        self.data_num = None
        self.feature_num = None
        self.featrue_selected = None

    def fit(self, x, y, metrics):
        assert len(x) == len(y)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
        self.data_num, self.feature_num = x_train.shape

        population = []
        swarm = []
        ts = []
        perf_list = []
        opti_perf_list = []
        t0 = clock()
        for t in range(self.max_iter):
            r = self.r0
            n = self.n0
            # initialization and mutation
            if self.local_search:
                if t == 0:
                    for j in range(n):
                        swarm.append(Particle(r, self.feature_num, self.k))
                else:
                    swarm = []
                    for i in range(int(0.9 * n)):
                        if len(population) == 1:
                            selected = np.array([0, 0], dtype=int)
                        else:
                            selected = np.random.choice(len(population), 2, replace=False)
                        p1 = population[selected[0]]
                        p2 = population[selected[1]]
                        offspring = Particle(r, self.feature_num, self.k)

                        ch = crossover(p1.position, p2.position, 10)
                        mutation_ga(ch, 0.1, self.k)

                        offspring.position = ch
                        offspring.feature_selected = binary2indices(ch)

                        swarm.append(offspring)

                    if n > len(swarm):
                        len_swarm = len(swarm)
                        swarm += [Particle(r, self.feature_num, self.k) for _ in range(n - len_swarm)]
            else:
                swarm = []
                for j in range(n):
                    swarm.append(Particle(r, self.feature_num, self.k))

            # (n, r) inner loop
            while r <= self.data_num and n >= 1:
                n_samples = np.random.permutation(self.data_num)[:r]
                for particle in swarm:
                    particle.resource = r  # update the resource of the particle
                    particle.evaluate(x=[x_train[n_samples], x_valid],
                                      y=[y_train[n_samples], y_valid],
                                      clf=self.clf,
                                      metrics=metrics)
                swarm.sort(reverse=True)
                succesful_parition = math.ceil(n / self.eta)
                # failure_swarm += swarm[succesful_parition:]

                swarm = swarm[:succesful_parition]

                if r == self.data_num or succesful_parition == 1:
                    break
                r = int(r * self.eta)
                if r > self.data_num:
                    r = self.data_num
                n = succesful_parition

            print("t = ", t, "best perf:", swarm[0].performance)
            if len(population) == 0:
                population = swarm
            else:
                len_popu = self.population_size
                population += swarm
                population.sort(reverse=True)
                population = population[:len_popu]
            perfs = [p.performance for p in population]
            print("len population:", len(population), "perf:", perfs)
            ts.append(clock() - t0)
            perf_list.append(swarm[0].performance)
            opti_perf_list.append(population[0].performance)

        self.featrue_selected = binary2indices(population[0].position)

        return ts, perf_list, opti_perf_list

    def transform(self, x):
        return x[:, self.featrue_selected]

if __name__ == '__main__':
    indices = [5, 4, 0, 7, 8]
    binary = indices2binary(indices, 10)
    print(binary)

    p = Particle(100, 10, 3)
    print(p.position.__class__)
    print(p.feature_selected)
