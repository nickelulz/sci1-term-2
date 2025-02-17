from itertools import permutations
import random
import numpy as np
from coins import CoinFlipEngine

from util import generate_random_distribution, generate_random_markov

def generate_random_model(size_range=(1,2), memory_depth_range=(1,1)):
    """
    Generates a random CoinFlipEngine model with random weights.
    """
    size = random.randint(size_range[0], size_range[1])
    probabilities_matrix = np.array([generate_random_distribution(size) for n in range(size)]).reshape(size, size)

    memory_depth = random.randint(memory_depth_range[0], memory_depth_range[1])
    get_random_coin = lambda: random.randint(0, size)

    # generate the cartesian product to get all possible combinations
    all_possible_output_combinations = list(product(range(size), repeat=memory_depth))

    markov = [] 
    for coin in range(size):
        coin_markov_ht = HashTable(len(all_possible_output_combinations))
        for output_sequence in all_possible_output_combinations:
            coin_markov_ht.insert(output_sequence, get_random_coin())

    return CoinFlipEngine(
        probabilities = probabilities_matrix,
        markov = markov,
        memory_depth = memory_depth,
        size = size 
    )
