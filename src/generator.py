from itertools import permutations, product
import random
import numpy as np
from coins import CoinFlipEngine

def generate_random_distribution(n):
    """
    Generates a random distribution of n outputs that sums to 100,
    ensuring no zero values.
    """
    if n > 100:
        raise ValueError("Cannot ensure nonzero values if n > 100")

    # Generate random values and ensure a minimum allocation per value
    random_values = np.random.rand(n)  
    min_value = 0.5  # Smallest fraction to ensure nonzero allocation
    scaled_values = random_values + min_value  # Shift all values up
    normalized_values = scaled_values / scaled_values.sum()  # Normalize

    # Scale and round to integers
    distribution = (normalized_values * 100).round().astype(int)

    # Fix rounding errors to ensure sum is exactly 100
    diff = 100 - distribution.sum()
    for _ in range(abs(diff)):
        index = np.random.choice(n)
        distribution[index] += 1 if diff > 0 else -1

    # lazy solution, but whatever
    if np.sum(distribution) != 100 or 0 in distribution:
        return generate_random_distribution(n)
    else:
        return distribution

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

def generate_random_model(size_range=(1,2), memory_depth_range=(1,1), benchmark=False, benchmark_flips=int(1e4)):
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
        size = size,
        benchmark = benchmark,
        benchmark_flips = benchmark_flips
    )

def markov_reaches_all(markovs: list[dict], size: int):
    reachable_coins = ()

    for markov in markovs:
        reachable_coins = reachable_coins + tuple(markov.values())

    reachable_coins = set(reachable_coins)
    return len(reachable_coins) == size

def generate_all_possible_markov_system(size: int, memory_depth: int) -> list[list[dict]]:
    """
    Generates a list of all of the possible markov lists that could exist given a
    size and a memory depth.  Used for reverse engineering by Freakbob.
    """
    combinations = list(product(range(size), repeat=memory_depth))
    
    # Generate all possible transition mappings for a single Markov dictionary
    all_possible_markov_single = list(product(range(size), repeat=len(combinations)))
    
    # Convert tuples to dictionary format
    all_possible_markov_dicts = [
        {comb: mapping[i] for i, comb in enumerate(combinations)}
        for mapping in all_possible_markov_single
    ]
    
    # Generate all possible ways to assign Markov dicts to `size` coins
    all_possible_markov_systems = list(product(all_possible_markov_dicts, repeat=size))

    valid_markovs = []
    for system in all_possible_markov_systems:
        system = list(system)
        if markov_reaches_all(system, size):
            valid_markovs.append(system)

    return valid_markovs 
