#!/usr/bin/env python

"""
Example shows how with statement works

The example shows three approaches
1. manual approach: try / finally block
2. classic class approach
3. contextlib.contextmanager approach

Paradigm:
setup code
try:
    user code
finally:
    tear down (clean up)
"""


from contextlib import contextmanager
import logging


def my_function():
    logging.debug('First debug data')
    logging.error('Error data')
    logging.debug('Second debug data')


# standard class way
class debug_logging_class(object):
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        self.logger = logging.getLogger()
        self.old_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(self.level)
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


@contextmanager
def debug_logging(level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


if __name__ == '__main__':
    logging.warning('\n\n------ default -----')
    my_function()

    logging.warning('\n\n----- manual try -----')
    my_logger = logging.getLogger()
    my_old_level = my_logger.getEffectiveLevel()
    my_logger.setLevel(logging.DEBUG)
    try:
        my_function()
    finally:
        my_logger.setLevel(my_old_level)

    logging.warning('\n\n------ contextmanager -----')
    with debug_logging(logging.DEBUG):
        my_function()

    logging.warning('\n\n----- with class -----')
    with debug_logging_class(logging.DEBUG):
        my_function()

    logging.warning('\n\n----- after all -----')
    my_function()