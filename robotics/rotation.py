#!/usr/bin/env python

"""
Basic SO3 rotation
"""

from pylab import *


def RotX(a: float):
    return array([[1,      0,      0],
                  [0, cos(a), -sin(a)],
                  [0, sin(a),  cos(a)]])


def RotY(a: float):
    return array([[cos(a),  0, sin(a)],
                  [     0,  1,     0],
                  [-sin(a), 0, cos(a)]])

def RotZ(a: float):
    return array([[cos(a), -sin(a), 0],
                  [sin(a),  cos(a), 0],
                  [    0 ,      0 , 1]])


def GetQuarternion():
    """ Get Quarternion representation
    """
    pass


def Identity():
    return eye(3)


def Inverse():
    """ Compute rotation inverse
    """
    print("TBD")
    pass


def RotZYX(z: float, y: float, x: float):
    return RotZ(z) * RotY(y) * RotX(x)


def SLERP():
    print("TBD")
    pass
