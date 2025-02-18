import sys
sys.path.insert(1,'../src')
from coins import CoinFlipEngine
from util import interpret_simple_markov
from hash_table import HashTable
import numpy as np

DEFAULT_COIN = CoinFlipEngine(
    probabilities = np.array([[50, 50], [50, 50]]),
    markov = interpret_simple_markov(np.array([[1, 0], [1, 0]])),
    size = 2,
    name = 'Default Coin'
)

NON_COIN = CoinFlipEngine(
    probabilities = np.array([[100]]),
    markov = interpret_simple_markov(np.array([[0]])),
    size = 1,
    name = 'Non Coin'
)

SIMPLE_MARKOV_1 = CoinFlipEngine(
    probabilities = np.array([[90, 10], [50, 50]]),
    markov = interpret_simple_markov(np.array([[0, 1], [1, 0]])),
    size = 2,
    name = 'Simple Markov 1'
)

SIMPLE_MARKOV_2 = CoinFlipEngine(
    probabilities = np.array([[10, 90], [90, 10]]),
    markov = interpret_simple_markov(np.array([[1, 0], [1, 0]])),
    size = 2,
    name = 'Simple Markov 2'
)

SIMPLE_MARKOV_3 = CoinFlipEngine(
    probabilities = np.array([[90, 10], [10, 90]]),
    markov = interpret_simple_markov(np.array([[1, 0], [1, 0]])),
    size = 2,
    name = 'Simple Markov 3'
)

# n=2 coin with memory=3
MARKOV_MEMORY_1 = HashTable(8)
MARKOV_MEMORY_1.insert((0,0,0), 1)
MARKOV_MEMORY_1.insert((0,0,1), 0)
MARKOV_MEMORY_1.insert((0,1,0), 1)
MARKOV_MEMORY_1.insert((0,1,1), 1)
MARKOV_MEMORY_1.insert((1,0,0), 0)
MARKOV_MEMORY_1.insert((1,0,1), 1)
MARKOV_MEMORY_1.insert((1,1,0), 1)
MARKOV_MEMORY_1.insert((1,1,1), 0)

MARKOV_MEMORY_3_COIN = CoinFlipEngine(
    probabilities = np.array([[30, 70], [60, 40]]),
    markov = [ MARKOV_MEMORY_1, MARKOV_MEMORY_1 ],
    size = 2,
    memory_depth = 3,
    name = 'Markov 2x2 with Memory 3'
)

ALL_COINS = [ DEFAULT_COIN, NON_COIN, SIMPLE_MARKOV_1, SIMPLE_MARKOV_2, SIMPLE_MARKOV_3, MARKOV_MEMORY_3_COIN ]
