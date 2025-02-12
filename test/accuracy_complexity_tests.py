import unittest
from unittest import TestCase

import sys
sys.path.insert(1,'../src')
from coins import *
from benchmarker import * 

from coin_examples import *

class AccuracyComplexityTests(TestCase):
    def same_coin_accuracy_test(self):

def testAccuracy(revCoin, dataProbs, flips=int(1e4)):
  n = 10 #arbitraty number of times we test it
  accuracies = []
  for i in range(n):
    res = perform_coin_flips(revCoin, flips)
    accuracies.append(max(res["empirical_probabilities"] - dataProbs))
  return np.mean(accuracies)
