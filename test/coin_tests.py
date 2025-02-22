import unittest
from unittest import TestCase
from tabulate import tabulate
from coin_examples import *

import sys
sys.path.insert(1,'../src')
from coins import *
from flips import *
from reverse_engineer import reverse_engineer_model
from error import calculate_model_error_all

class CoinTests(TestCase):
    """
    Stats for All Coins
    """
    def test_coin_stats_all(self):
        header = [ 'Name', 'Complexity', 'Variance', 'Theoretical', 'Standing', 'Empirical', 'Calix', 'Calix Error', 'Sleepr', 'Sleeper Error' ]
        table = []

        for coin in ALL_COINS:
            standing_dist = ''
            if coin.size == 2:
                standing_dist = np.round(calculate_standing_distribution_2d(coin.probabilities), 1)

            row = [ coin.name, coin.complexity, coin.variance, coin.theoretical_distribution,
                    standing_dist ]

            guessed_coins = reverse_engineer_model(coin.benchmark_result.flip_history, 
                                                   coin, benchmark=True, debug=True)
            errors = calculate_model_error_all(coin, guessed_coins)
            errors = np.round(errors, 3) 
    
            row.append(coin.empirical_distribution)

            # Calix
            row.append(guessed_coins[0].empirical_distribution)
            row.append(errors[0])

            # Sleeper 
            row.append(guessed_coins[1].empirical_distribution)
            row.append(errors[1])

            table.append(row)

        
        table.insert(0, header)
        print()
        print(tabulate(table, tablefmt='grid'))

    def standing_dist_coin(self, coin):
        dist = calculate_standing_distribution_2d(coin.probabilities)
        print(coin.name, 'standing:', np.round(dist, 1), 'theoretical:', coin.theoretical_distribution) 

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
        self.assert_equal_arrays(theo_dist, emp_dist, 0.5)

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

    """
    Markov chains with higher memory
    """

    def test_markov_memory_3(self):
        coin = MARKOV_MEMORY_3_COIN
        result = perform_coin_flips(coin, int(1e4))
        print()
        print(coin.name)
        print('empirical distribution:', result.empirical_distribution)
        print('sequence prob hist:')
        histogram = evaluate_sequence_probability_history(coin.size, result.flip_history, memory_depth=3)
        # print(histogram)
        for sequence, entry in histogram.items():
            print(sequence, 'count', entry['count'], 'percent', entry['percentage'])

if __name__ == '__main__':
    unittest.main()
