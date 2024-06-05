import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

from VRPGraph import VRPGraph


class GeneticAlgorithmCVRP:
    def __init__(self, graph: VRPGraph, population_size: int, num_generations: int, mutation_rate: float,
                 vehicle_capacity: float):
        self.graph = graph
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.vehicle_capacity = vehicle_capacity
        self.depot = 0

        # Initialize population
        self.population = self.initialize_population()

    def initialize_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.population_size):
            route = list(range(1, self.graph.num_nodes))
            random.shuffle(route)
            population.append(route)
        return population

    def fitness(self, individual: List[int]) -> float:
        total_distance = 0.0
        vehicle_load = 0.0
        previous_node = self.depot

        for node in individual:
            demand = self.graph.graph.nodes[node]['demand'][0]
            if vehicle_load + demand > self.vehicle_capacity:
                total_distance += self.graph.euclid_distance(previous_node, self.depot)
                previous_node = self.depot
                vehicle_load = 0.0

            total_distance += self.graph.euclid_distance(previous_node, node)
            vehicle_load += demand
            previous_node = node

        total_distance += self.graph.euclid_distance(previous_node, self.depot)
        return total_distance

    def select(self) -> List[int]:
        tournament_size = 5
        selected = random.sample(self.population, tournament_size)
        selected.sort(key=lambda ind: self.fitness(ind))
        return selected[0]

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        child = [-1] * size

        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]

        pointer = end
        for node in parent2:
            if node not in child:
                if pointer >= size:
                    pointer = 0
                while child[pointer] != -1:
                    pointer += 1
                    if pointer >= size:
                        pointer = 0
                child[pointer] = node

        return child

    def mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve(self):
        new_population = []
        for _ in range(self.population_size):
            parent1 = self.select()
            parent2 = self.select()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self) -> Tuple[List[int], float]:
        best_fitness = float('inf')
        best_route = None

        for generation in range(self.num_generations):
            self.evolve()
            current_best_individual = min(self.population, key=lambda ind: self.fitness(ind))
            current_best_fitness = self.fitness(current_best_individual)

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_route = self.build_complete_route(current_best_individual)

            print(f"Generation {generation + 1}/{self.num_generations} completed - Best Fitness: {best_fitness}")

        return best_route, best_fitness

    def build_complete_route(self, individual: List[int]) -> List[int]:
        complete_route = []
        current_route = []
        vehicle_load = 0.0

        for node in individual:
            demand = self.graph.graph.nodes[node]['demand'][0]
            if vehicle_load + demand > self.vehicle_capacity:
                complete_route.append(0)  # Return to depot
                complete_route.extend(current_route)
                complete_route.append(0)  # Start new route from depot
                current_route = []
                vehicle_load = 0.0

            current_route.append(node)
            vehicle_load += demand

        if current_route:
            complete_route.append(0)  # Return to depot
            complete_route.extend(current_route)
            complete_route.append(0)  # End at depot

        return complete_route

    def plot_route(self, best_route: List[int]):
        pos = nx.get_node_attributes(self.graph.graph, "coordinates")
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        self.graph.set_default_node_attributes()
        self.graph.draw(ax)

        # Define colors for different routes
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_idx = 0

        # Plot the best route with different colors for each sub-route
        sub_route = []
        for i in range(1, len(best_route)):
            if best_route[i] == 0:
                # Draw sub-route
                for j in range(len(sub_route) - 1):
                    nx.draw_networkx_edges(
                        self.graph.graph,
                        pos,
                        edgelist=[(sub_route[j], sub_route[j + 1])],
                        edge_color=colors[color_idx % len(colors)],
                        ax=ax,
                        width=2.0,
                    )
                if sub_route:
                    nx.draw_networkx_edges(
                        self.graph.graph,
                        pos,
                        edgelist=[(0, sub_route[0]), (sub_route[-1], 0)],
                        edge_color=colors[color_idx % len(colors)],
                        ax=ax,
                        width=2.0,
                    )
                sub_route = []
                color_idx += 1
            else:
                sub_route.append(best_route[i])

        # Draw the last sub-route if exists
        if sub_route:
            for j in range(len(sub_route) - 1):
                nx.draw_networkx_edges(
                    self.graph.graph,
                    pos,
                    edgelist=[(sub_route[j], sub_route[j + 1])],
                    edge_color=colors[color_idx % len(colors)],
                    ax=ax,
                    width=2.0,
                )
            if sub_route:
                nx.draw_networkx_edges(
                    self.graph.graph,
                    pos,
                    edgelist=[(0, sub_route[0]), (sub_route[-1], 0)],
                    edge_color=colors[color_idx % len(colors)],
                    ax=ax,
                    width=2.0,
                )

        # Draw the nodes with labels
        node_labels = {node: str(node) for node in self.graph.graph.nodes()}
        label_pos = {node: (pos[node][0], pos[node][1] + 0.03) for node in self.graph.graph.nodes()}
        nx.draw_networkx_labels(self.graph.graph, label_pos, labels=node_labels, font_size=10, ax=ax)

        self.graph.draw(ax)
        plt.title('Best Route Found by Genetic Algorithm')
        plt.show()


# Example usage:
num_nodes = 20
num_depots = 1
plot_demand = False

# Create VRP graph
graph = VRPGraph(num_nodes, num_depots, plot_demand)

# Genetic algorithm parameters
population_size = 100
num_generations = 500
mutation_rate = 0.02
vehicle_capacity = 1.0  # Vehicle capacity is 1

# Run genetic algorithm
ga = GeneticAlgorithmCVRP(graph, population_size, num_generations, mutation_rate, vehicle_capacity)
best_route, total_distance = ga.run()

print("Best route:", best_route)
print("Total distance:", total_distance)

# Plot the best route
ga.plot_route(best_route)
