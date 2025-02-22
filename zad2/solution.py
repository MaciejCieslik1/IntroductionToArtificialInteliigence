# Author: Maciej Cieślik

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable


class Chromosome:
    def __init__(self, length: int, array: List[int] = None):  # if array is None it should be initialized with random binary vector
        if array is None:
            self.array = np.random.randint(2, size=length)
        else:
            self.array = array

    def decode(self, lower_bound: int, upper_bound: int, aoi: Tuple[float, float]) -> float:
        chromosome_part = self.array[lower_bound:upper_bound]
        value = 0
        max_possible_value = 0
        min_possible_value = 0
        multiplier = len(chromosome_part) - 1
        for gene in chromosome_part:
            value += gene * 2 ** multiplier
            max_possible_value += 2 ** multiplier
            multiplier -= 1
        return min_max_norm(value, min_possible_value, max_possible_value, aoi[0], aoi[1])

    def mutation(self, probability: float) -> None:
        if np.random.rand() < probability:
            choosen_index = random.randint(0, len(self.array) - 1)
            self.array[choosen_index] = 1 - self.array[choosen_index]

    def crossover(self, other: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        division_index = random.randint(1, len(self.array) - 1)
        new_self_array = np.concatenate((self.array[:division_index], other.array[division_index:]))
        new_other_array = np.concatenate((other.array[:division_index], self.array[division_index:]))
        return Chromosome(len(new_self_array), new_self_array), Chromosome(len(new_other_array), new_other_array)


class GeneticAlgorithm:
    def __init__(self, chromosome_length: int, obj_func_num_args: int, objective_function: Callable[..., float], aoi: Tuple[float, float],
                population_size: int = 1000, tournament_size: int = 2, mutation_probability: float = 0.05,
                crossover_probability: float = 0.8, num_steps: int = 30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_lengths = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

        self.population = []
        self.population = [Chromosome(chromosome_length) for _ in range(population_size)]

    def eval_objective_func(self, chromosome: 'Chromosome') -> dict[str, float]:
        variables_values = {}
        for i in range(self.obj_func_num_args):
            decoded_value = chromosome.decode(int(i * self.bits_per_arg), int((i + 1) * self.bits_per_arg), self.aoi)
            current_variable = f"x{i + 1}"
            variables_values[current_variable] = decoded_value
        variables_values["value"] = self.objective_function(variables_values)
        return variables_values

    def tournament_selection(self) -> 'Chromosome':
        choosen_chromosomes = random.sample(self.population, self.tournament_size)
        winner = choosen_chromosomes[0]
        for current_chromosome in choosen_chromosomes:
            if self.eval_objective_func(current_chromosome)["value"] < self.eval_objective_func(winner)["value"]:
                winner = current_chromosome
        return winner

    def reproduce(self, parents: Tuple['Chromosome', 'Chromosome']) -> Tuple['Chromosome', 'Chromosome']:
        if np.random.rand() < self.crossover_probability:
            child1, child2 = parents[0].crossover(parents[1])
        else:
            child1 = parents[0]
            child2 = parents[1]
        child1.mutation(self.mutation_probability)
        child2.mutation(self.mutation_probability)
        return child1, child2

    def plot_func(self, trace: List[dict], filename: str):
        X = np.arange(-2, 3, 0.05)
        Y = np.arange(-4, 2, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
        plt.figure()
        plt.contour(X, Y, Z, 10)
        x1_values = []
        x2_values = []
        values = []
        for point_info in trace:
            x1_values.append(point_info["x1"])
            x2_values.append(point_info["x2"])
            values.append(point_info["value"])
        min_value = min(values)
        max_value = max(values)
        cmaps = [[1 - (value - min_value) / (max_value - min_value), 0, 0] for value in values]
        plt.scatter(x1_values, x2_values, c=cmaps, s=10)
        plt.title("Genetic algorithm in function minimalization")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-2, 3)
        plt.ylim(-4, 2)
        plt.grid()
        plt.savefig(filename)
        plt.show()

    def run(self) -> dict[str, float]:
        step = 0
        winners = []
        new_population = []
        trace = []
        while step < self.num_steps:
            winners = [self.tournament_selection() for _ in range(self.population_size)]
            for index in range(0, self.population_size // 2, 2):
                child1, child2 = self.reproduce((winners[index], winners[index + 1]))
                new_population.append(child1)
                new_population.append(child2)
            best_chromosome = new_population[0]
            best_chromosome_info = self.eval_objective_func(best_chromosome)
            for chromosome in new_population:
                chromosome_info = self.eval_objective_func(chromosome)
                if chromosome_info["value"] < best_chromosome_info["value"]:
                    best_chromosome = chromosome
                    best_chromosome_info = chromosome_info
            trace.append(best_chromosome_info)
            self.population = new_population
            step += 1
        return trace


def min_max_norm(val: int, min_val: int, max_val: int, new_min: int, new_max: int) -> float:
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


def f_x(variables_values: dict[str, float]) -> float:
    power_index1 = 0
    power_index2 = 0
    for key, value in variables_values.items():
        power_index1 -= value ** 2
        if key == "x1":
            power_index2 -= (value - 1) ** 2
        elif key == "x2":
            power_index2 -= (value + 2) ** 2
        else:
            power_index2 -= value ** 2
    return 1.5 - np.exp(power_index1) - 0.5 * np.exp(power_index2)


def display_result(trace: List[dict]):
    for index, point in enumerate(trace):
        current_value = point["value"]
        print(f"Step {index + 1}, value: {current_value}")


algorithm = GeneticAlgorithm(20, 2, f_x, (-3, 3), 500, 2, 0.05, 0.8, 30)
trace = algorithm.run()
display_result(trace)
algorithm.plot_func(trace, "test.png")
