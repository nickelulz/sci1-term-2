import random
import numpy as np
from coins import CoinFlipEngine

from util import generate_random_distribution

def generate_random_model(max_size=2, min_size=1):
    """
    Generates a random CoinFlipEngine model with pseudorandom weights.
    """

    size = random.randint(max_size, max_size)
    probabilities_matrix = np.array([generate_random_distribution(size) for n in range(size)]).reshape(size, size)
    markov_matrix = np.array([n for n in range(size)] * size).reshape(size, size)

    return CoinFlipEngine(
        probabilities = probabilities_matrix,
        markov = markov_matrix,
        size = size 
    )
