import random
import numpy as np
import matplotlib.pyplot as plt
from VRPGraph import VRPGraph  # Ensure the correct module name here

class GeneticAlgorithmVRP:
    def __init__(self, vrp_graph: VRPGraph, population_size: int = 100, generations: int = 500, mutation_rate: float = 0.01):
        self.vrp_graph = vrp_graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        """Initialize population with random routes"""
        population = []
        for _ in range(self.population_size):
            route = np.random.permutation(self.vrp_graph.num_nodes)
            population.append(route)
        return population

    def fitness(self, route):
        """Calculate the fitness of a route based on total distance and demand"""
        total_distance = 0
        total_demand = 0
        for i in range(len(route) - 1):
            total_distance += self.vrp_graph.euclid_distance(route[i], route[i + 1])
            total_demand += self.vrp_graph.graph.nodes[route[i]]["demand"]
        total_demand += self.vrp_graph.graph.nodes[route[-1]]["demand"]
        return total_distance, total_demand

    def select(self):
        """Select parents for crossover using tournament selection"""
        selected = random.choices(self.population, k=2)
        return selected[0] if self.fitness(selected[0])[0] < self.fitness(selected[1])[0] else selected[1]

    def crossover(self, parent1, parent2):
        """Perform ordered crossover"""
        start, end = sorted(random.sample(range(self.vrp_graph.num_nodes), 2))
        child = [-1] * self.vrp_graph.num_nodes
        child[start:end] = parent1[start:end]
        pointer = end
        for node in parent2:
            if node not in child:
                if pointer >= len(child):
                    pointer = 0
                child[pointer] = node
                pointer += 1
        return child

    def mutate(self, route):
        """Perform swap mutation"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve(self):
        """Evolve the population through generations"""
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select()
                parent2 = self.select()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
            best_route = min(self.population, key=self.fitness)
            print(f"Generation {generation} | Best fitness: {self.fitness(best_route)[0]}")

        best_route = min(self.population, key=self.fitness)
        return best_route, self.fitness(best_route)

# Example usage
vrp_graph = VRPGraph(num_nodes=20, num_depots=1, plot_demand=True)  # Adjust num_depots if necessary
ga_vrp = GeneticAlgorithmVRP(vrp_graph)
best_route, best_fitness = ga_vrp.evolve()

total_distance, total_demand = best_fitness
print(f"Best route: {best_route}")
print(f"Total distance: {total_distance}")
print(f"Total demand satisfied: {total_demand}")

# Optional: Plotting the final solution
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(best_route) - 1):
    vrp_graph.visit_edge(best_route[i], best_route[i + 1])
vrp_graph.draw(ax)
plt.show()
