from itertools import permutations, product
import random
import numpy as np
from coins import CoinFlipEngine

def generate_random_distribution(n):
    """
    Generates a random distribution of n outputs that sums to 100.
    """
    random_values = np.random.rand(n) 
    normalized_values = random_values / random_values.sum()
    distribution = (normalized_values * 100).round().astype(int)

    diff = 100 - distribution.sum()
    for _ in range(abs(diff)):
        index = np.random.choice(n)
        distribution[index] += 1 if diff > 0 else -1

    return np.array(distribution.tolist())

def generate_possible_combinations(size: int, memory_depth: int) -> list[tuple]:
    return list(product(range(size), repeat=memory_depth))

def generate_random_markov_single(size: int, memory_depth: int, combinations: list[tuple]) -> dict:
    num_combos = len(combinations)
    associated_coins = list(range(0, size)) + random.choices(range(0, size-1), k = num_combos - size)

    coin_markov_dict = {}
    for i, output_sequence in enumerate(combinations):
        coin_markov_dict[output_sequence] = associated_coins[i]

    return coin_markov_dict

def generate_random_markov(size: int, memory_depth: int) -> list[dict]:
    combinations = generate_possible_combinations(size, memory_depth) 
    return [ generate_random_markov_single(size, memory_depth, combinations) for coin in range(size) ] 

def generate_random_model(size_range=(1,2), memory_depth_range=(1,1)):
    """
    Generates a random CoinFlipEngine model with random weights.
    """
    size = random.randint(size_range[0], size_range[1]) 
    memory_depth = random.randint(memory_depth_range[0], memory_depth_range[1])

    probabilities_matrix = np.array([generate_random_distribution(size) for n in range(size)]).reshape(size, size)
    markov = generate_random_markov(size, memory_depth)

    return CoinFlipEngine(
        probabilities = probabilities_matrix,
        markov = markov,
        memory_depth = memory_depth,
        size = size 
    )
