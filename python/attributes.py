#!/usr/bin/env python

from __future__ import print_function


class BaseClass(object):
    def __init__(self):
        self.public_attribute = 1.0
        self._protected_attribute = 2.0
        self.__private_attribute = 3.0


if __name__ == '__main__':
    print('attributes')

    bc = BaseClass()
    print('Public    attribute: %f' % bc.public_attribute)
    # PyCharm shows a warning
    print('Protected attribute: %f' % bc._protected_attribute)

    try:
        print('Private   attribute: %f' % bc.__private_attribute)
    except AttributeError:
        print('Failed to access private attribute, AttributeError')

    # Hack to access private attribute
    # PyCharm shows a warning
    print('Private attribute (HACK): %f' % bc._BaseClass__private_attribute)
