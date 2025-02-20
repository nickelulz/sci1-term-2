import numpy as np
import sys
from random import random

from util import *
from flips import *

def calculate_standing_distribution_2d(probabilities):
    """
    Calculates standing distribution for 2D coins
    """
    p = probabilities[0][0] / 100
    q = probabilities[1][0] / 100
    denom = (1 - p + q)

    if denom == 0:
        raise ValueError(f'returned zero: p {p}, q {q}')

    r = q / denom

    return np.array([r, 1-r]) * 100

def calculate_theoretical_distribution(probabilities) -> np.array:
    """
    Calculates the theoretical probability distribution
    of the coin pool given its Markov Chain and input
    probability distribution matricies
    """
    M_power_40 = np.linalg.matrix_power(np.array(probabilities) * 0.01, 40)

    # rounded to the nearest 0.1
    return np.round((M_power_40 * 100)[0], 1)

def calculate_model_complexity(system) -> np.float64:
    """
    Calculates the complexity of a CoinFlipEngine model based on size, variance, and memory
    """
    return round(system.memory_depth ** 2 * system.variance * system.size, 1)

def calculate_variance(system) -> np.float64:
    """
    Calculates the variance for a full coin system by calculating
    the variance of each coin individually and summing them up, as defined
    as the deviance from the mean (which is 100% divided by number of outputs,
    i.e. the difference from this coin to its fully unbiased cousin)
    """
    total = 0
    for coin_prob in (system.probabilities / 100):
        coin_variance = np.sum(np.abs(coin_prob - 1 / system.number_of_outputs))
        total += coin_variance
    return round(total, 1)

class CoinFlipEngine():
    def __init__(self, probabilities: np.array, markov: np.array, size: int, 
                 memory_depth = 1, initial_coin_index = 0, name = None, 
                 benchmark = False, benchmark_flips=int(1e4)):
        self.size = size
        self.number_of_coins = size
        self.number_of_outputs = size
        self.name = name

        self.markov = markov
        self.probabilities = np.array(probabilities)
        self.thresholds = [weights_to_ranges(probability_array) for probability_array in probabilities]

        self.initial_coin_index = initial_coin_index
        self.memory_depth = memory_depth

        # Markov State 
        self.current_coin_index = initial_coin_index
        self.memory = []

        if benchmark:
            self.benchmark(benchmark_flips)

    def benchmark(self, flips) -> CoinFlipResult:
        self.theoretical_distribution = calculate_theoretical_distribution(self.probabilities)
        self.variance = calculate_variance(self)
        self.complexity = calculate_model_complexity(self)

        self.benchmark_result = perform_coin_flips(self, flips)
        self.empirical_distribution = self.benchmark_result.empirical_distribution
        self.sequence_histogram = evaluate_sequence_probability_history(self.number_of_outputs, 
                                                                        self.benchmark_result.flip_history, 
                                                                        self.memory_depth)
        return self.benchmark_result

    def reset_markov(self) -> None:
        self.current_coin_index = self.initial_coin_index
        self.memory = []

    def flip(self, print_on_switch=False, output=sys.stdout) -> int:
        """
        Uses the markov and weight information to produce a random coin flip
        Returns the output and stores the next coin in the markov chain
        """
        coin_thresholds = self.thresholds[self.current_coin_index]
        random_number = random()

        for output_index, threshold in enumerate(coin_thresholds):
            # This is the output
            if random_number <= threshold:
                self.memory.append(output_index)

                # choose next coin
                if len(self.memory) == self.memory_depth:
                    if not (tuple(self.memory) in self.markov[self.current_coin_index]):
                        print(self.size, self.memory, self.memory_depth, self.markov)
                    next_coin = self.markov[self.current_coin_index][tuple(self.memory)]

                    if print_on_switch:
                        print(f'Coin Decision: {self.current_coin_index} -> {next_coin} due to output {self.memory}', file=output)

                    self.current_coin_index = next_coin
                    self.memory = []

                return output_index

        # did not hit a threshold
        print('Error! Did not hit a threshold')
        print(random_number, coin_thresholds, self.probabilities[self.current_coin_index])
        print(self.probabilities)
        print(set(self.markov[self.current_coin_index].values()))

    def __str__(self) -> str:
        return (
            f'CoinFlipEngine(\n' +
            f'probabilities={self.probabilities},\n' +
            f'markov={self.markov},\n' +
            f'number_of_coins={self.number_of_coins},\n' +
            f'number_of_outputs={self.number_of_outputs}\n' +
            f'theoretical_distribution={self.theoretical_distribution})')
