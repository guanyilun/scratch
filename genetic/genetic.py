#%%
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

class GeneticAlgorithm(ABC):

    @abstractmethod
    def init_population(self, n_pop): ...

    @abstractmethod
    def evaluate_fitness(self, child): ...

    @abstractmethod
    def crossover(self, p1, p2): ...

    @abstractmethod
    def mutate(self, child): ...
    
    @abstractmethod
    def is_converged(self, fitness): ...

    def w_func(self, fitness):
        return fitness

    def build_mating_pool(self, n_pop, population, fitness, w_func=None):
        w_func = w_func or self.w_func
        return np.random.choice(
            len(population),
            (n_pop, 2), 
            p=w_func(fitness),
            replace=True
        )

    def run(self, max_gen=100, n_pop=100, population=None, top_n=5):
        if population is None:
            population = self.init_population(n_pop)
        fitness = np.array(list(map(self.evaluate_fitness, population)))
        i_gen = 0
        while (not self.is_converged(fitness)) and (i_gen < max_gen):
            i_gen += 1
            new_population = []
            mating_pool = self.build_mating_pool(n_pop, population, fitness)
            for (p1, p2) in tqdm(mating_pool):
                try:
                    child = self.crossover(population[p1], population[p2])
                    child = self.mutate(child)
                except:
                    continue
                new_population.append(child)
            fitness = np.array(list(map(self.evaluate_fitness, new_population)))
            population = new_population
            print(f"Generation: {i_gen}, Best: {population[np.argmax(fitness)]}, Fitness: {np.max(fitness)}")

        # return best individuals
        return [(population[i], fitness[i]) for i in np.argsort(fitness)[-top_n:]]

#%%
# import random, string, math

# class HelloWorld(GeneticAlgorithm):

#     def __init__(self, target, mutation_rate, **kwargs):
#         super().__init__(**kwargs)
#         self.target = target
#         self.mutation_rate = mutation_rate
#         self.w_func = lambda f: f**2
        
#     def init_population(self, n_pop):
#         population = []
#         for _ in range(n_pop):
#             individual = random.choices(string.ascii_letters + string.digits, k=len(self.target))
#             population.append(individual)
#         return population
    
#     def evaluate_fitness(self, child):
#         fitness = 0
#         for i in range(len(child)):
#             if child[i] == self.target[i]:
#                 fitness += 1
#         return fitness
    
#     def crossover(self, p1, p2):
#         midpoint = math.floor(len(p1))
#         return p1[:midpoint] + p2[midpoint:]
    
#     def mutate(self, child):
#         child = child.copy()
#         prob = np.random.uniform(0, 1, len(child))
#         for i in range(len(child)):
#             if prob[i] < self.mutation_rate:
#                 child[i] = random.choice(string.ascii_letters + string.digits)
#         return child
    
#     def is_converged(self, fitness):
#         return max(fitness) == len(self.target)

# if __name__ == "__main__":
#     HelloWorld(1000, "HelloWorld", 0.01).run(1000)
