#!/usr/bin/env python
from primes import is_prime
import unittest

class PrimesTestCase(unittest.TestCase):
    def test_is_five_prime(self):
        self.assertTrue(is_prime(5))

    def test_is_four_no_prime(self):
        self.assertFalse(is_prime(4), msg='Four is not prime!')

    # edge case
    def test_is_zero_not_prime(self):
        self.assertFalse(is_prime(0), msg='Zero is not prime')

    def test_negative_number(self):
        for index in range(-1, -10, -1):
            self.assertFalse(is_prime(index), msg='Negative number is not prime') 

if __name__ == '__main__':
    unittest.main()
