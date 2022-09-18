"""
Visualize Microbial Genetic Algorithm to find the maximum point in a graph.
Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
from random import sample
import matplotlib.pyplot as plt


class MGA_XAXB:
    def __init__(self, dna_size, population_size, cross_rate, mutation_rate, ans=None, dna_bank_range=(0, 10)):
        self.dna_size = dna_size  # 一個XAXB遊戲有幾個數字
        self.population_size = population_size  # 總共有幾筆XAXB資料同時在現場做配對
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

        if dna_bank_range[1] - dna_bank_range[0] < dna_size:
            raise ValueError("DNA基因庫的範圍需要更大，或是一組DNA的基因數要更小")

        self.dna_data_bank = list(range(*dna_bank_range))  # DNA資料庫，XAXB的可填數字範圍
        self.population = np.array(
            [sample(self.dna_data_bank, dna_size) for _ in range(population_size)]  # 建立人群，建立XAXB資料
        )

        self.ans = sample(self.dna_data_bank, dna_size) if ans is None else ans

    def get_fitness(self):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            score = 0
            compare = self.population[i] == self.ans

            # compare_True_idx = np.argwhere(compare == True)
            # compare_True_idx = compare_True_idx.reshape(compare_True_idx.shape[0])
            compare_False_idx = np.argwhere(compare == False)
            compare_False_idx = compare_False_idx.reshape(compare_False_idx.shape[0])

            # fitness[i] += compare_True_idx.shape[0] * 3
            fitness[i] += np.sum(compare) * 3
            for idx in compare_False_idx:
                if self.population[i, idx] in self.ans:
                    fitness[i] += 1

        return fitness

    def cross_over(self, fitness: np.ndarray = None):
        if fitness is None:
            fitness = self.get_fitness()

        # _idxes = np.random.choice(self.population_size, size=2, replace=False)
        _idxes = sample(list(range(self.population_size)), 2)
        # good_bad_idx = np.zeros(2)
        if fitness[_idxes[0]] > fitness[_idxes[1]]:
            good_bad_idx = _idxes
        else:
            good_bad_idx = _idxes[::-1]

        change_idx = np.random.random(size=self.dna_size) > self.cross_rate

        for i, do_change in enumerate(change_idx):
            if not do_change:
                continue

            # 判斷要交換的值是否已經出現在bad裡面
            if self.population[good_bad_idx[0], i] in self.population[good_bad_idx[1]]:
                #  good要交換時發現bad裡面有相同值，就讓bad的那兩項互換即可
                #  要交換的bad裡面的目標位置是i，發現bad裡面的重複的位置是a，最後要讓a和i交換
                bad_a_idx = np.where(self.population[good_bad_idx[1]] == self.population[good_bad_idx[0], i])[0]

                self.population[good_bad_idx[1], [bad_a_idx]], self.population[good_bad_idx[1], [i]] = \
                    self.population[good_bad_idx[1], [i]], self.population[good_bad_idx[1], [bad_a_idx]]
            else:
                self.population[good_bad_idx[1], i] = self.population[good_bad_idx[0], i]

        self.mutation(good_bad_idx[1])

    def mutation(self, pop_idx):
        mutation_idxes = np.random.random(self.dna_size) < self.mutation_rate
        if not (True in mutation_idxes):
            return

        m_dna_bank = []
        # 清除已經存在的DNA
        for i in self.dna_data_bank:
            if not (i in self.population[pop_idx]):
                m_dna_bank.append(i)

        for _idx, mutate in enumerate(mutation_idxes):

            if np.random.random() > 0.5 or len(m_dna_bank) == 0:
                # 把值跟列表內其他的值互換, a_idx和_idx交換
                idxes = list(range(self.dna_size))
                a_idx = sample(idxes, 1)
                while a_idx == _idx:
                    a_idx = sample(idxes, 1)

                self.population[pop_idx, a_idx], self.population[pop_idx, _idx] = self.population[pop_idx, _idx], self.population[pop_idx, a_idx]

            else:
                # 把值做改變

                if mutate:
                    self.population[pop_idx, _idx] = m_dna_bank.pop(
                        sample(list(range(len(m_dna_bank))), 1)[0]
                    )

    def evolve(self, limit_times):
        FULL_SCORE = self.dna_size * 3

        fitness = self.get_fitness()
        count = 0
        y = []
        while fitness[fitness.argsort()[-1]] != FULL_SCORE:
            fitness = self.get_fitness()
            self.cross_over(fitness)
            # y.append(fitness.sum())
            y.append(fitness[fitness.argsort()[-1]])
            count += 1

            if count % 500 == 0:
                print("目前是第", count, "代")

            if count >= limit_times:
                print("進化未完成，超過額定上限 ", limit_times, " 次")
                break

        print("在第", count, "代完成進化\n")
        x = list(range(count))
        return x, y, count


def test_():
    DNA_SIZE = 10  # DNA length；要猜幾個數字
    POP_SIZE = 20  # population size
    CROSS_RATE = 0.6  # mating probability (DNA crossover)
    MUTATION_RATE = 0.02  # mutation probability
    N_GENERATIONS_LIMIT = 20000

    xaxb = MGA_XAXB(DNA_SIZE, POP_SIZE, CROSS_RATE, MUTATION_RATE, dna_bank_range=(0, 30))  # dna_bank：有甚麼數字可以拿來猜，(0, 30) 則是0~29這些數字可以被猜
    print(xaxb.ans)
    x, y, _ = xaxb.evolve(N_GENERATIONS_LIMIT)

    print(xaxb.population[xaxb.get_fitness().argsort()[-1]])

    plt.subplot(2, 1, 1)
    # plt.xlim(0, N_GENERATIONS)
    # plt.ylim(0, DNA_SIZE * 3)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':

    # for i in range(10):
    test_()
