#!/usr/bin/env python3

"""
Show how to use variable args
1) positional variable args
2) keyword variable args
3) all variable args
"""


def test_var_positional_args(p_arg, *args):
    print('p_arg = ' + p_arg)
    for arg in args:
        print('another arg = ' + arg)


def test_var_keyboard_args(**kwargs):
    print('Keyword: ', kwargs)


def test_var_all_args(*args, **kwargs):
    print('Position: ', args)
    print('Keyword : ', kwargs)


if __name__ == '__main__':
    test_var_positional_args('hello')
    test_var_positional_args('hello', 'va1', 'va2', 'va3')

    test_var_keyboard_args(va1='variable argument 1',
                           va2='variable argument 2')

    try:
        test_var_keyboard_args(1, 2, 3)  # raises TypeError
    except TypeError:
        print('test_var_keyward_args takes 0 positional args, but 3 were given')

    test_var_all_args(1, 2, 3,
                      va1='variable argument 1',
                      va2='variable argument 2')






