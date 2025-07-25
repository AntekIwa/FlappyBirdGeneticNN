import numpy as np
import random
import copy

class GeneticTrainer:
    def __init__(self, network, population_size, mutation_rate, elite_fraction):
        #setting population_size, mutation_rate, elite_fraction and generating random networks for our population
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.population = [self.random_clone(network) for _ in range(population_size)]

    def random_clone(self, net): #generate random NN
        clone = copy.deepcopy(net)
        for i in range(len(clone.weights)):
            clone.weights[i] += np.random.randn(*clone.weights[i].shape) * 0.5
            clone.biases[i] += np.random.randn(*clone.biases[i].shape) * 0.5
        return clone

    def evaluate_population(self,  evaluate_func): #fitness function to checking how good is our network
        return [evaluate_func(net) for net in self.population]

    def evolve(self, fitness_scores):
        #we are choosing best NN (sorted by fitness) and then crossovering them and mutate
        sorted_indicies = np.argsort(fitness_scores)[::-1]
        elite_size = int(self.elite_fraction * self.population_size)
        elite = [self.population[sorted_indicies[i]] for i in range(elite_size)]

        new_population = elite.copy()
        while len(new_population) < self.population_size: #lets add some new NN which are crossovers of our elite NNs
            A, B = random.sample(elite, 2)
            child = self._crossover(A, B)
            self._mutate(child)
            new_population.append(child)
        self.population = new_population

    def _crossover(self, A, B):
        child = copy.deepcopy(A)
        for i in range(len(child.weights)):
            maskW = np.random.rand(*child.weights[i].shape) < 0.5 #generating random mask which show us which weighs or biases we are chosing from each NN
            maskB = np.random.rand(*child.biases[i].shape) < 0.5
            child.weights[i] = np.where(maskW, A.weights[i], B.weights[i])
            child.biases[i] = np.where(maskB, A.biases[i], B.biases[i])
        return child

    def _mutate(self, net):
        for i in range(len(net.weights)):
            weight_mask = np.random.rand(*net.weights[i].shape) < self.mutation_rate #random bitmask to choose parameters to mutate
            bias_mask = np.random.rand(*net.biases[i].shape) < self.mutation_rate
            net.weights[i] += weight_mask * np.random.randn(*net.weights[i].shape)
            net.biases[i] += bias_mask * np.random.randn(*net.biases[i].shape)
