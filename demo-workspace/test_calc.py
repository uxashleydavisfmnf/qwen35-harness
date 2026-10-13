"""Tests for calc.py.  Run with: python test_calc.py"""

import unittest

from calc import add, mean, multiply


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)

    def test_mean(self):
        self.assertEqual(mean([2, 4, 6]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
