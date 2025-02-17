import unittest
from unittest import TestCase

from coin_examples import *

import sys
sys.path.insert(1,'../src')
from coins import *
from flips import *

class CoinTests(TestCase):
    """
    Helpers
    """
    def ensure_numpy_array(self, arr):
        return arr if isinstance(arr, np.ndarray) else np.array(arr)

    def assert_equal_arrays(self, actual, expected, tolerance):
        self.assertTrue(np.allclose(actual, expected, atol=tolerance),
                        f'Expected: {expected}, Got: {actual}')


    """
    Theoretical Distribution Calculation Tests
    """

    def theo_dist_calc(self, probabilities, expected):
        # sanitize input
        probabilities = self.ensure_numpy_array(probabilities)
        expected = self.ensure_numpy_array(expected)

        theo_dist = calculate_theoretical_distribution(probabilities)
        self.assert_equal_arrays(theo_dist, expected, 0.1)        

    def theo_dist_coin(self, coin, expected):
        self.theo_dist_calc(coin.probabilities, expected) 

    def test_theo_dist_1(self):
        self.theo_dist_calc([[50, 50],[50, 50]], [50, 50])

    def standing_dist_coin(self, coin):
        dist = calculate_standing_distribution_2d(coin.probabilities)
        print(coin.name, 'standing:', dist, 'theoretical:', coin.theoretical_distribution)
    
    def test_standing_dist(self):
        print()
        for coin in filter(lambda coin: coin.size == 2, ALL_COINS):
            self.standing_dist_coin(coin)

    """
    Direct Coin Tests
    """

    def test_theo_dist_default_coin(self):
        self.theo_dist_coin(DEFAULT_COIN, [50, 50])

    def test_theo_dist_non_coin(self):
        self.theo_dist_coin(NON_COIN, [100])

    def test_theo_dist_simple_markov_1(self):
        self.theo_dist_coin(SIMPLE_MARKOV_1, [83.3, 16.7])

    def test_theo_dist_simple_markov_2(self):
        self.theo_dist_coin(SIMPLE_MARKOV_2, [50, 50])

    def test_theo_dist_simple_markov_3(self):
        self.theo_dist_coin(SIMPLE_MARKOV_3, [50, 50])

    """
    Convergence Tests
    """
    def coin_conv(self, coin):
        theo_dist = calculate_theoretical_distribution(coin.probabilities)
        emp_dist = perform_coin_flips(coin, int(1e6)).empirical_distribution
        self.assert_equal_arrays(theo_dist, emp_dist, 0.1)

    def test_coin_conv_default(self):
        self.coin_conv(DEFAULT_COIN)

    def test_coin_conv_non_coin(self):
        self.coin_conv(NON_COIN)

    def test_coin_conv_simple_markov_1(self):
        self.coin_conv(SIMPLE_MARKOV_1)

    def test_coin_conv_simple_markov_2(self):
        self.coin_conv(SIMPLE_MARKOV_2)

    def test_coin_conv_simple_markov_3(self):
        self.coin_conv(SIMPLE_MARKOV_3)

if __name__ == '__main__':
    unittest.main()
