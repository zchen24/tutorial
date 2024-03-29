#!/usr/bin/env python
""" Shows how to parse_args

1) positional argument
    $ parse_args.py a1

2) optional argument
    $ parse_args.py a1 -o 4

3) conflicting arguments
Example:
    $ parse_args.py a1 -g1 -g2
    > error: argument -g2: not allowed with argument -g1
"""

from __future__ import print_function
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parsing example')

    # positional argument
    parser.add_argument('a1', help='position arg 1')
    parser.add_argument('a2', nargs="?", help='position arg 2')

    # optional argument
    parser.add_argument('-o', '--optional', type=int,
                        help='optional argument')

    parser.add_argument('-j', '--joints', nargs="+", help='optional arg array +')
    parser.add_argument('-o0', nargs="*", help='optional arg array *')

    # conflicting group
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-g1', action='store_true', help='arg 1 in exclusive group')
    group.add_argument('-g2', action='store_true', help='arg 2 in exclusive group')

    # parse it!
    args = parser.parse_args()

    # print args
    print('positional a1 = ', args.a1)
    print('positional a2 = ', args.a2)
    print('optional arg = ', args.optional)
    print('optional joints = ', args.joints)
    print('optional o0 = ', args.o0)
    print('group arg g1 = ', args.g1)
    print('group arg g2 = ', args.g2)
