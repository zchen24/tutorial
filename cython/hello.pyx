"""
Hello World Cython pyx
"""

from libc cimport math

def say_hello_to(name):
    print("Hello {}!".format(name))
