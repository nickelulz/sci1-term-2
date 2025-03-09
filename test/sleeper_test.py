import unittest
from unittest import TestCase
from coin_examples import *

import sys
sys.path.insert(1, '../src')
from sleeper import Sleeper
from error import calculate_model_error

class SleeperTest(TestCase):
    def test_sleeper(self):
        coin = MARKOV_MEMORY_3_COIN
        guess = Sleeper(coin.benchmark_result.flip_history, coin.memory_depth, coin.size, True)
        error = calculate_model_error(coin, guess)
        print('Sleeper error: ', error)


if __name__ == '__main__':
    unittest.main()
