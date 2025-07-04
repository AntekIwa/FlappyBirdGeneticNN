import numpy as np
from neural_network import NeuralNetwork
from genetic import GeneticTrainer
from FlappyEnv import FlappyEnvPopulation

INPUT_SIZE = 9
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

POPULATION_SIZE = 100
GENERATIONS = 250

template_net = NeuralNetwork(
    layers_sizes=[INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE],
    activations=["identity","sigmoid"],
    loss_name="mse"
)

trainer = GeneticTrainer(
    network=template_net,
    population_size=POPULATION_SIZE,
    mutation_rate=0.05,
    elite_fraction=0.2
)

for g in range(GENERATIONS):
    print(f"\n=== Generacja {g+1}/{GENERATIONS} ===")
    env = FlappyEnvPopulation(trainer.population, render=True, gen = g + 1)
    fitnesses = env.run()
    avg_fit = sum(fitnesses) / len(fitnesses)
    best_fit = max(fitnesses)
    print(f"Najlepszy fitness: {best_fit:.1f} | Średni: {avg_fit:.1f}")

    trainer.evolve(fitnesses)

print("\nPrezentacja najlepszej populacji:")
viewer = FlappyEnvPopulation(trainer.population, render=True)
viewer.run()
