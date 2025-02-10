import unittest
from unittest import TestCase

import sys
sys.path.insert(1,'../src')
from coins import *
from benchmarker import * 

from coin_examples import *

class AccuracyComplexityTests(TestCase):
    def same_coin_accuracy_test(self):
