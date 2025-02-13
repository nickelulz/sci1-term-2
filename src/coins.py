from util import *
import numpy as np
import sys
import random

def calculate_theoretical_distribution(markov, probabilities) -> np.array:
    """
    Calculates the theoretical probability distribution
    of the coin pool given its Markov Chain and input
    probability distribution matricies
    """
    if len(markov) == 1:
        return probabilities[0]
    else:
        M_power_40 = np.linalg.matrix_power(np.array(probabilities) * 0.01, 40)

        # rounded to the nearest 0.1
        return np.round((M_power_40 * 100)[0], 1)

def calculate_model_complexity(system) -> np.float64:
    """
    Calculates the complexity of a CoinFlipEngine model based on size, variance, and memory
    """
    return (system.memory_depth - 1) + system.variance * system.size

def calculate_variance(system) -> np.float64:
    '''
    Calculates the variance of a coin system
    (1) https://en.wikipedia.org/wiki/Variance
    # Solve the steady-state equation and normalize
    modifiers = system.probabilities.transpose() - np.eye(system.number_of_outputs) 
    modifiers[-1] = np.ones(system.number_of_outputs)

    # Right-hand side of the equation and normalize
    dependents = np.zeros(system.number_of_outputs)
    dependents[-1] = 1

    # Solve the linear system
    stationary_dist = np.linalg.lstsq(modifiers, dependents, rcond=None)[0]
    mean_probability = np.sum(stationary_dist) / len(stationary_dist)

    # Biased estimate
    return np.mean((stationary_dist - mean_probability) ** 2) 
    '''
    #variance for one coin
    return np.sum(system.probabilities[0] - (100/len(system.probabilities[0])))

class CoinFlipEngine():
    def __init__(self, probabilities, markov, size, initial_coin_index=0, memory_depth=1):
        # only working with square matricies
        self.size = size
        self.number_of_coins = size
        self.number_of_outputs = size

        self.markov = markov
        self.probabilities = probabilities
        self.thresholds = [weights_to_ranges(probability_array) for probability_array in probabilities]

        self.initial_coin_index = initial_coin_index
        self.memory_depth = memory_depth

        # Markov State 
        self.current_coin_index = initial_coin_index
        self.memory = []

    def benchmark(self):
        self.theoretical_distribution = calculate_theoretical_distribution(markov, probabilities)
        self.variance = calculate_variance(self)
        self.complexity = calculate_model_complexity(markov, probabilities)
        self.model_convergence_benchmark, self.empirical_distribution = benchmark_model_convergence(self)

    def reset_markov(self):
        self.current_coin_index = self.initial_coin_index
        self.memory = []

    def flip(self, print_on_switch=False, output=sys.stdout) -> int:
        """
        Uses the markov and weight information to produce a random coin flip
        Returns the output and stores the next coin in the markov chain
        """
        coin_thresholds = self.thresholds[self.current_coin_index]
        random_number = random.random()

        for output_index, threshold in enumerate(coin_thresholds):
            if random_number < threshold:
                if print_on_switch:
                    print(f'Coin Decision: {self.current_coin_index} -> ' + 
                          '{self.markov[self.current_coin_index][output_index]} due to output {output_index}', file=output)
                self.memory.append(output_index)
                self.current_coin_index = self.markov[self.current_coin_index][output_index]
                return output_index

    def __str__(self) -> str:
        return (
            f'CoinFlipEngine(\n' +
            f'probabilities={self.probabilities},\n' +
            f'markov={self.markov},\n' +
            f'number_of_coins={self.number_of_coins},\n' +
            f'number_of_outputs={self.number_of_outputs}\n' +
            f'theoretical_distribution={self.theoretical_distribution})')
