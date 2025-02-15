import sys
sys.path.insert(1,'../src')
from coins import CoinFlipEngine
import numpy as np

DEFAULT_COIN = CoinFlipEngine(
    probabilities = np.array([[50, 50], [50, 50]]),
    markov = np.array([[1, 0], [1, 0]]),
    size = 2,
    name = 'Default Coin'
)

NON_COIN = CoinFlipEngine(
    probabilities = np.array([[100]]),
    markov = np.array([[0]]),
    size = 1,
    name = 'Non Coin'
)

SIMPLE_MARKOV_1 = CoinFlipEngine(
    probabilities = np.array([[90, 10], [50, 50]]),
    markov = np.array([[0, 1], [1, 0]]),
    size = 2,
    name = 'Simple Markov 1'
)

SIMPLE_MARKOV_2 = CoinFlipEngine(
    probabilities = np.array([[10, 90], [90, 10]]),
    markov = np.array([[1, 0], [1, 0]]),
    size = 2,
    name = 'Simple Markov 2'
)

SIMPLE_MARKOV_3 = CoinFlipEngine(
    probabilities = np.array([[90, 10], [10, 90]]),
    markov = np.array([[1, 0], [1, 0]]),
    size = 2,
    name = 'Simple Markov 3'
)

ALL_COINS = [ DEFAULT_COIN, NON_COIN, SIMPLE_MARKOV_1, SIMPLE_MARKOV_2, SIMPLE_MARKOV_3 ]
