from dataclasses import dataclass
import numpy as np
import sys
from random import random

@dataclass
class CoinFlipResult:
    count_distribution: np.array
    empirical_distribution: np.array
    flip_history: np.array
    percent_history: np.array

def perform_coin_flips(coin, number_of_flips, print_on_switch=False, output_file=sys.stdout):
    count_distribution = [0] * coin.number_of_outputs
    percent_data = [[0] for _ in range(coin.number_of_outputs)]
    flip_data = []

    for flips in range(1, number_of_flips + 1):
        result_output = coin.flip(print_on_switch, output_file)
        flip_data.append(result_output)        
        count_distribution[result_output] += 1

        for output in range(coin.number_of_outputs):
            percent_data[output].append(count_distribution[output] / flips)

    empirical_probabilities = (np.array(count_distribution) / number_of_flips) * 100
    coin.reset_markov()

    return CoinFlipResult(count_distribution, empirical_probabilities, flip_data, np.array(percent_data) * 100)

def evaluate_sequence_probability_history(num_flips, num_outputs, flip_history, memory_depth=2):
    # Dictionary to store sequence occurrences over time
    sequence_counts = defaultdict(int)
    sequence_history = defaultdict(list)

    for i in range(num_flips - memory_depth + 1):
        sequence = tuple(flip_history[i:i + memory_depth])
        sequence_counts[sequence] += 1

        # Calculate probabilities for each sequence
        for a in range(num_outputs):
            for b in range(num_outputs):
                key = (a, b)
                count = sequence_counts.get(key, 0)
                sequence_history[key].append(count / (i + 1))

    # Convert sequences to string format for better readability
    formatted_history = {"-".join(map(str, key)): values for key, values in sequence_history.items()}

    return formatted_history
