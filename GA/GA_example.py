import numpy as np
import matplotlib.pyplot as plt



DNA_SIZE = 10
POP_SIZE = 100
CROSS_RATE = 0.8  # 有80%的人口進行交叉配對，剩下的人用隨機?
MUTATION_RATE = 0.003
N_GENERATIONS = 200 # 要遺傳幾代
X_BOUND = [0, 5]


def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


def get_fitness(pred):
    """判斷適應度(分數"""
    return pred + 1e-3 - np.min(pred)


def translateDNA(pop):
    """翻譯DNA資訊"""
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):
    """適者生存，選取父母基因"""
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):
    """基因的交叉配對"""
    i_ = np.random.randint(POP_SIZE, size=1)
    cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(bool)
    parent[cross_points] = pop[i_, cross_points]
    return parent

def mutate(child):
    """基因變異"""
    for point in range(DNA_SIZE):
        if MUTATION_RATE > np.random.random():
            child[point] = 1 if child[point] == 0 else 0
    return child


def main():
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA

    plt.ion()  # something about plotting
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))
    F_values = 0

    for _ in range(N_GENERATIONS):
        F_values = F(translateDNA(pop))  # compute function value by extracting DNA

        # something about plotting
        # if 'sca' in globals(): sca.remove()
        # sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5);
        # plt.pause(0.05)

        # GA part (evolution)
        fitness = get_fitness(F_values)
        print("Most fitted DNA: ", pop[np.argmax(fitness), :])
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child  # parent is replaced by its child

    # plt.ioff()
    plt.scatter(translateDNA(pop), F_values, s=10, lw=0, c='red', alpha=0.5)
    plt.show()


main()
