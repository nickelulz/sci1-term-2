from collections import defaultdict

import sys
sys.path.insert(1,'../src')
from coins import CoinFlipEngine
from util import interpret_simple_markov
import numpy as np

DEFAULT_2X2_MARKOV = interpret_simple_markov(
        np.array([[0, 1], 
                  [0, 1]]))

DEFAULT_COIN = CoinFlipEngine(
    probabilities = np.array([[50, 50], 
                              [50, 50]]),
    markov = DEFAULT_2X2_MARKOV,
    size = 2,
    name = 'Default Coin',
    benchmark = True
)

NON_COIN = CoinFlipEngine(
    probabilities = np.array([[100]]),
    markov = interpret_simple_markov(
        np.array([[0]])),
    size = 1,
    name = 'Non Coin',
    benchmark = True
)

SIMPLE_MARKOV_1 = CoinFlipEngine(
    probabilities = np.array([[90, 10], 
                              [50, 50]]),
    markov = interpret_simple_markov(
        np.array([[0, 1], 
                  [1, 0]])),
    size = 2,
    name = 'Simple Markov 1',
    benchmark = True
)

SIMPLE_MARKOV_2 = CoinFlipEngine(
    probabilities = np.array([[10, 90], 
                              [90, 10]]),
    markov = interpret_simple_markov(
        np.array([[0, 1], 
                  [0, 1]])),
    size = 2,
    name = 'Simple Markov 2',
    benchmark = True
)

SIMPLE_MARKOV_3 = CoinFlipEngine(
    probabilities = np.array([[90, 10], 
                              [10, 90]]),
    markov = interpret_simple_markov(
        np.array([[0, 1], 
                  [0, 1]])),
    size = 2,
    name = 'Simple Markov 3',
    benchmark = True
)

# n=2 coin with memory=3
MARKOV_MEMORY_1 = defaultdict(int) 
MARKOV_MEMORY_1[(0,0,0)] = 1
MARKOV_MEMORY_1[(0,0,1)] = 0
MARKOV_MEMORY_1[(0,1,0)] = 1
MARKOV_MEMORY_1[(0,1,1)] = 1
MARKOV_MEMORY_1[(1,0,0)] = 0
MARKOV_MEMORY_1[(1,0,1)] = 1
MARKOV_MEMORY_1[(1,1,0)] = 1
MARKOV_MEMORY_1[(1,1,1)] = 0

MARKOV_MEMORY_3_COIN = CoinFlipEngine(
    probabilities = np.array([[30, 70], 
                              [60, 40]]),
    markov = [ MARKOV_MEMORY_1, MARKOV_MEMORY_1 ],
    size = 2,
    memory_depth = 3,
    name = 'Markov 2x2 with Memory 3',
    benchmark = True
)

ALL_COINS = [ DEFAULT_COIN, NON_COIN, SIMPLE_MARKOV_1, SIMPLE_MARKOV_2, SIMPLE_MARKOV_3, MARKOV_MEMORY_3_COIN ]
