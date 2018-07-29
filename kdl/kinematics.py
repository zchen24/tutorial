#!/usr/bin/env python

"""
Demonstrate basic robot kinematics operations using KDL
 - forward kinematics
 - inverse kinematics
 - jacobian
 - inverse jacobian (IKSolverVel)

Copyright 2018 Zihan Chen
"""


from PyKDL import *
from math import pi


def make_puma560():
    robot = Chain()
    robot.addSegment(Segment())
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.0, pi/2, 0.0, 0.0),
                             RigidBodyInertia(0, Vector().Zero(), RotationalInertia(0, 0.35, 0, 0, 0, 0))))
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.4318, 0.0, 0.0, 0.0),
                             RigidBodyInertia(17.4, Vector(-.3638, .006, .2275),
                                              RotationalInertia(0.13, 0.524, 0.539, 0, 0, 0))))
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.0203, -pi/2, 0.15005, 0.0),
                             RigidBodyInertia(4.8, Vector(-.0203, -.0141, .070),
                                              RotationalInertia(0.066, 0.086, 0.0125, 0, 0, 0))))
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.0, pi/2, 0.4318, 0.0),
                             RigidBodyInertia(0.82, Vector(0, .019, 0),
                                              RotationalInertia(1.8e-3, 1.3e-3, 1.8e-3, 0, 0, 0))))
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.0, -pi/2, 0.0, 0.0),
                             RigidBodyInertia(0.34, Vector().Zero(),
                                              RotationalInertia(.3e-3, .4e-3, .3e-3, 0, 0, 0))))
    robot.addSegment(Segment(Joint(Joint.RotZ),
                             Frame().DH(0.0, 0.0, 0.0, 0.0),
                             RigidBodyInertia(0.09, Vector(0, 0, .032),
                                              RotationalInertia(.15e-3, 0.15e-3, .04e-3, 0, 0, 0))))
    return robot


if __name__ == '__main__':
    print('creating a robot example')
    p560 = make_puma560()

    qz = JntArray(p560.getNrOfJoints())

    fk_solver = ChainFkSolverPos_recursive(p560)
    jac_solver = ChainJntToJacSolver(p560)
    ik_vel_solver = ChainIkSolverVel_pinv(p560)
    ik_solver = ChainIkSolverPos_NR(p560, fk_solver, ik_vel_solver)

    # forward kinematics
    t_ee = Frame()
    fk_solver.JntToCart(qz, t_ee)

    # jacobian
    jac = Jacobian(p560.getNrOfJoints())
    jac_solver.JntToJac(qz, jac)

    # inverse Jacobian
    #  There are a lot of inverse Jacobian solvers in KDL,
    #  here, pseudo-inverse is used
    xdot = Twist(Vector(0.1, 0, 0), Vector(0, 0, 0))
    qdot = JntArray(p560.getNrOfJoints())
    ik_vel_solver.CartToJnt(qz, xdot, qdot)

    # inverse kinematics
    qout = JntArray(p560.getNrOfJoints())
    qrand = JntArray(p560.getNrOfJoints())
    qrand[0] = 0.01
    qrand[1] = -0.015
    qrand[2] = 0.004
    ik_solver.CartToJnt(qz, t_ee, qout)
