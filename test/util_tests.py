import unittest
from unittest import TestCase

from random import randint
import numpy as np

import sys
sys.path.insert(1,'../src')
from util import *
from hash_table import HashTable

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

    # also ruthlessly stolen from geeksforgeeks
    def test_hash_table_basic(self):
        ht = HashTable(5) 

        ht.insert("apple", 3) 
        ht.insert("banana", 2) 
        ht.insert("cherry", 5) 

        self.assertTrue("apple" in ht) 
        self.assertFalse("durian" in ht)

        self.assertEqual(ht.search("banana"), 2) 

        ht.insert("banana", 4) 
        self.assertEqual(ht.search("banana"), 4) 

        ht.remove("apple") 
        self.assertEqual(len(ht), 2) 

    def test_hash_table_markov_1x1(self):
        markov_ht = HashTable(1)
        markov_ht.insert((0), 0)
        self.assertTrue((0) in markov_ht)
        self.assertFalse((0, 1) in markov_ht)
        self.assertFalse((1) in markov_ht)
        self.assertEqual(len(markov_ht), 1)
        self.assertEqual(markov_ht.search((0)), 0)

    def test_markov_interpreter(self):
        markov_arr = np.array([1, 0])
        markov_ht = HashTable(2)
        markov_ht.insert((0,), 1)
        markov_ht.insert((1,), 0)
        interpreted = interpret_individual_markov(markov_arr)
        self.assertEqual(interpreted.keys(), markov_ht.keys())

if __name__ == '__main__':
    unittest.main()
