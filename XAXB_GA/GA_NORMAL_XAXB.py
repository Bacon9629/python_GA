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
            compare_True_idx = np.argwhere(compare == True)
            compare_True_idx = compare_True_idx.reshape(np.prod(compare_True_idx.shape))
            compare_False_idx = np.argwhere(compare == False)
            compare_False_idx = compare_False_idx.reshape(np.prod(compare_False_idx.shape))

            score += compare_True_idx.shape[0] * 3
            for idx in compare_False_idx:
                if self.population[i, idx] in self.ans:
                    score += 1

            fitness[i] = score

        return fitness

    def cross_over(self, fitness: np.ndarray = None):
        if fitness is None:
            fitness = self.get_fitness()

        def _cross(dad, mom):
            child = mom.copy()
            p = np.random.random(self.dna_size) > 0.5
            child[p] = dad[p]
            return child

        _parent_amount = self.population_size // 4
        temp = fitness.argsort()
        parent_idxes = temp[-_parent_amount:]
        child_idxes =  temp[:-_parent_amount]

        for child_idx in child_idxes:
            dad_mom_idx = sample(list(parent_idxes), 2)
            self.population[child_idx] = _cross(*self.population[dad_mom_idx])
            self.mutation(child_idx)

    def mutation(self, pop_idx):
        mutation_idx = np.random.random(self.dna_size) < self.mutation_rate
        if not (True in mutation_idx):
            return

        m_dna_bank = []
        # 清除已經存在的DNA
        for i in self.dna_data_bank:
            if not (i in self.population[pop_idx]):
                m_dna_bank.append(i)

        for _idx, mutate in enumerate(mutation_idx):

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
                print("已到上限，進化失敗")
                break

        print("在第", count, "代結束進化")
        x = list(range(count))
        return x, y, count

if __name__ == '__main__':

    DNA_SIZE = 10  # DNA length
    POP_SIZE = 20  # population size
    CROSS_RATE = 0.6  # mating probability (DNA crossover)
    MUTATION_RATE = 0.02  # mutation probability
    N_GENERATIONS_LIMIT = 10000

    xaxb = MGA_XAXB(DNA_SIZE, POP_SIZE, CROSS_RATE, MUTATION_RATE, dna_bank_range=(0, 20))
    print(xaxb.ans)
    x, y, _ = xaxb.evolve(N_GENERATIONS_LIMIT)

    print(xaxb.population[xaxb.get_fitness().argsort()[-1]])

    # plt.subplot(2, 1, 1)
    # plt.xlim(0, N_GENERATIONS)
    # plt.ylim(0, DNA_SIZE * 3)
    plt.plot(x, y)
    plt.show()

    # ============================================

    # DNA_SIZE = 10  # DNA length
    # POP_SIZE = 20  # population size
    # CROSS_RATE = 0.8  # mating probability (DNA crossover)
    # MUTATION_RATE = 0.02  # mutation probability
    # N_GENERATIONS_LIMIT = 10000
    #
    # xaxb = MGA_XAXB(DNA_SIZE, POP_SIZE, CROSS_RATE, MUTATION_RATE, dna_bank_range=(0, 20))
    # print(xaxb.ans)
    # x, y, _ = xaxb.evolve(N_GENERATIONS_LIMIT)
    #
    # print(xaxb.population[xaxb.get_fitness().argsort()[-1]])
    #
    # plt.subplot(2, 1, 2)
    # # plt.xlim(0, N_GENERATIONS)
    # # plt.ylim(0, DNA_SIZE * 3)
    # plt.plot(x, y)
    # plt.show()
