#!/usr/bin/env python

from __future__ import print_function

if __name__ == '__main__':

    a_fruit = {'apple', 'pear', 'banana', 'grape'}
    b_fruit = {'apple', 'grape', 'kiwi', 'bear'}

    a_fruit_set = set(a_fruit)
    b_fruit_set = set(b_fruit)

    fruit_intersection = a_fruit_set.intersection(b_fruit_set)
    print(fruit_intersection)
    print('Is a_fruit_set a subset of b_fruit_set? Answer: ',
          str(a_fruit_set.issubset(b_fruit_set)))
