import matplotlib.pyplot as plt

from MY_PID import PID
from my_airplane import Airplane
from test_control_object import Test
from MGA import MGA
import numpy as np
from random import sample


class PID_GA(MGA):
    def __init__(self, population_size, cross_rate: np.ndarray, mutation_rate, PID_rate, pid_controller: PID):
        super().__init__(3, population_size, cross_rate, mutation_rate)
        self.dna_bank = PID_rate
        self.pop_list = np.zeros([population_size, 3], dtype=float)
        self.pid_controller = pid_controller

        self.init_mutation()
        self.sample_time = 0.01

    def get_fitness(self):
        """
        取得目前世代的所有人的適應分數
        """
        return self.__get_pid_loss_list()

    def cross_over(self, fitness):
        """
        依適應分數做配種，留下好的種，把爛的種改掉
        """

        no_use_idx = np.zeros(self.population_size, dtype=int)

        for i in range(population_size // 4 * 3):
            _idxes = np.random.choice(np.arange(self.population_size), 2, replace=False)
            while _idxes[0] in no_use_idx or _idxes[1] in no_use_idx:
                _idxes = np.random.choice(np.arange(self.population_size), 2, replace=False)

            if fitness[_idxes[0]] < fitness[_idxes[1]]:
                good_idx, bad_idx = _idxes[:]
                print("", end="")
            else:
                bad_idx, good_idx = _idxes[:]
                print("", end="")

            no_use_idx[i] = bad_idx

            if self.mutation(bad_idx):
                return

            self.pop_list[bad_idx, ::] = self.pop_list[good_idx, ::]
            self.pop_list[bad_idx][0] += (np.random.random() - 0.5) * self.cross_rate[0]
            self.pop_list[bad_idx][1] += (np.random.random() - 0.5) * self.cross_rate[1]
            self.pop_list[bad_idx][2] += (np.random.random() - 0.5) * self.cross_rate[2]


    def mutation(self, pop_idx):
        """
        變異
        """
        if np.random.random() < self.mutation_rate:
            self.pop_list[pop_idx, 0] = np.random.random() * (self.dna_bank[0, 1] - self.dna_bank[0, 0]) + self.dna_bank[0, 0]
            self.pop_list[pop_idx, 1] = np.random.random() * (self.dna_bank[1, 1] - self.dna_bank[1, 0]) + self.dna_bank[1, 0]
            self.pop_list[pop_idx, 2] = np.random.random() * (self.dna_bank[2, 1] - self.dna_bank[2, 0]) + self.dna_bank[2, 0]
            return True
        return False

    def init_mutation(self):
        for index in range(population_size):
            self.pop_list[index, 0] = np.random.random() * (self.dna_bank[0, 1] - self.dna_bank[0, 0]) + self.dna_bank[0, 0]
            self.pop_list[index, 1] = np.random.random() * (self.dna_bank[1, 1] - self.dna_bank[1, 0]) + self.dna_bank[1, 0]
            self.pop_list[index, 2] = np.random.random() * (self.dna_bank[2, 1] - self.dna_bank[2, 0]) + self.dna_bank[2, 0]

    def evolve(self, generation_amount):
        """
        執行世代交換的地方
        """
        fitness_history_list = np.zeros(generation_amount, dtype=float)
        fitness = np.zeros(self.population_size, dtype=float)
        for i in range(generation_amount):
            fitness = self.get_fitness()
            self.cross_over(fitness)
            fitness_history_list[i] = np.min(fitness)
            if i % 200 == 0:
                print("generation: ", i, "  best loss: ", fitness_history_list[i])

        return fitness_history_list, self.pop_list[np.argmin(fitness)]

    def __get_pid_loss_list(self):
        loss_list = np.zeros(self.population_size, dtype=float)
        for index, item in enumerate(self.pop_list):
            self.pid_controller.clear_and_init(*item)
            self.pid_controller.simulate()

            loss_list[index] = self.pid_controller.get_loss()

        return loss_list

    def __get_control_object(self):
        return Airplane(self.sample_time)


if __name__ == '__main__':

    def F(x):

        return -(x*x*x - 2 * x*x)

        # if x / 2 - x // 2 == 0.5:
        #     return 5
        # else:
        #     return 0

        # if x < 0.5:
        #     return 0
        # elif x < 1:
        #     return 5
        # elif x < 1.5:
        #     return 0
        # else:
        #     return 5

    population_size = 100
    cross_rate = np.array([10, 5, 0])
    mutation_rate = 0.1
    generation_amount = 400
    PID_RATE = np.array([[0, 10], [-5, 5], [-5, 5]])
    control_object = Airplane(0.01)
    # control_object = Test()

    pid_controller = PID(control_object, 0, 0, 0, F, 0.01, 0.2, 2)

    pid_ga = PID_GA(population_size, cross_rate, mutation_rate, PID_RATE, pid_controller)
    fitness_history_list, best_pid_value = pid_ga.evolve(generation_amount)
    print("PID: ", best_pid_value)

    pid_controller.clear_and_init(*best_pid_value)
    # pid_controller.clear_and_init(14.79994563, -94.37529224, 3.76736659)

    targets_y, my_values_y, t = pid_controller.simulate()
    print("loss: ", pid_controller.get_loss())

    plt.subplot(2, 1, 1)
    # plt.ylim(0, 1)
    plt.plot(fitness_history_list, color="red")
    plt.subplot(2, 1, 2)
    plt.plot(t, my_values_y, color="blue")
    plt.plot(t, targets_y, "--", color="red")

    plt.show()
