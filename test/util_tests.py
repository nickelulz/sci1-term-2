import unittest
from unittest import TestCase

from random import randint
import numpy as np

import sys
sys.path.insert(1,'../src')
from util import *
from generator import *

class UtilTests(TestCase):
    def w2r(self, input_arr, expected_arr):
        input_arr = np.array(input_arr)
        result = weights_to_ranges(input_arr)
        expected_arr = np.array(expected_arr)

        self.assertTrue(np.allclose(result, expected_arr, atol=1e-8), 
                        f'{input_arr} ~> {result}, not {expected_arr}')

    def test_w2r_3_output_1(self):
        self.w2r([20, 50, 30], [0.2, 0.7, 1.0])

    def test_w2r_3_output_2(self):
        self.w2r([50, 40, 10], [0.5, 0.9, 1.0])

    def test_w2r_1_output(self):
        self.w2r([100], [1.0])

    def test_w2r_5_output(self):
        self.w2r([10,  20,  30,  30,  10],
                 [0.1, 0.3, 0.6, 0.9, 1.0])

    def test_rand_dist_equals_1(self):
        for i in range(10):
            n = randint(1, 100) 
            dist = generate_random_distribution(n)
            dist_sum = np.sum(dist)
            self.assertEqual(dist_sum, 100, f'Distribution {dist}, sum {dist_sum} != 1')

    def test_interpret_markov_1(self):
        arr = np.array([[1, 0], [1, 0]])
        markov_interp = interpret_simple_markov(arr)
        markov = [
            { (0,): 1, (1,): 0 },
            { (0,): 1, (1,): 0 }
        ]
        self.assertEqual(markov_interp, markov)

if __name__ == '__main__':
    unittest.main()
