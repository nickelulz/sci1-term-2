from dataclasses import dataclass
import numpy as np
import sys
from random import random
from collections import defaultdict

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

    return CoinFlipResult(count_distribution, empirical_probabilities, np.array(flip_data), np.array(percent_data) * 100)

def evaluate_sequence_probability_history(num_outputs, flip_history, memory_depth=2) -> dict:
    sequence_counts = defaultdict(int)
    num_flips = len(flip_history)
    total_sequences = num_flips - memory_depth + 1
    
    for i in range(total_sequences):
        sequence = tuple(flip_history[i:i + memory_depth])
        sequence_counts[sequence] += 1
    
    sequence_histogram = {} 
    for sequence, count in sequence_counts.items():
        sequence_histogram[sequence] = {
            "count": count, 
            "percentage": round((count / total_sequences) * 100, 1)
        }
    
    return sequence_histogram
