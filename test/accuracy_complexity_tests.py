from tabulate import tabulate
import unittest
from unittest import TestCase

import sys
sys.path.insert(1,'../src')
from coins import *
from error import *

from coin_examples import *

class ErrorComplexityTests(TestCase):
    def relative_complexity_test(self, lesser, greater):
        self.assertTrue(lesser.complexity <= greater.complexity, f'ERR: Lesser is more complex than greater: {lesser.complexity} > {greater.complexity}')

    def test_complexity_relative(self):
        self.relative_complexity_test(NON_COIN, DEFAULT_COIN)
        self.relative_complexity_test(SIMPLE_MARKOV_1, SIMPLE_MARKOV_2)
        self.relative_complexity_test(SIMPLE_MARKOV_1, SIMPLE_MARKOV_3)
        self.relative_complexity_test(SIMPLE_MARKOV_2, SIMPLE_MARKOV_3)

    def test_same_coin_error(self):
        err = calculate_model_error(DEFAULT_COIN, DEFAULT_COIN)
        self.assertTrue(err == 0, f'Error calculation is incorrect! Returned: {err}') 

    def relative_error_test(self, coin_a, coin_b, coin_c):
        """
        Asserts that Coin A -> Coin b has less error than Coin A -> Coin C
        """
        error_ab = calculate_model_error(coin_a, coin_b)
        error_ac = calculate_model_error(coin_a, coin_c)
        self.assertTrue(error_ab < error_ac, f'Coin A is more similar to coin C than coin B: {error_ac} < {error_ab}')

    def test_different_coin_error_relative(self):
        self.relative_error_test(DEFAULT_COIN, SIMPLE_MARKOV_1, NON_COIN)
        self.relative_error_test(SIMPLE_MARKOV_1, SIMPLE_MARKOV_3, NON_COIN)
        self.relative_error_test(SIMPLE_MARKOV_2, SIMPLE_MARKOV_3, SIMPLE_MARKOV_1)

if __name__ == '__main__':
    unittest.main()
