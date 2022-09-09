
class MGA:
    def __init__(self, dna_size, population_size, cross_rate, mutation_rate):
        """
        dna_size : 一個人有幾個DNA
        population_size : 一個世代有幾個人
        cross_rate : 繁殖時每項DNA的交換機率
        mutation_rate : 每項DNA的變異機率
        """
        self.dna_size = dna_size
        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

    def get_fitness(self):
        """
        取得目前世代的所有人的適應分數
        """
        pass

    def cross_over(self, fitness):
        """
        依適應分數做配種，留下好的種，把爛的種改掉
        """
        pass

    def mutation(self, pop_idx):
        """
        變異
        """
        pass

    def evolve(self, generation_amount):
        """
        執行世代交換的地方
        """
        pass
