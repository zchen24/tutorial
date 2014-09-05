#include <iostream>
#include <cisstRobot.h>
#include <cisstVector.h>
#include <kdl/chain.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>


// Use DVRK MTM

using namespace KDL;

//standard  1.5708  0.0000 0.0 0.0000 revolute active -1.5708
//standard  0.0000  0.2794 0.0 0.0000 revolute active -1.5708
//standard -1.5708  0.3645 0.0 0.0000 revolute active  1.5708
//standard  1.5708  0.0000 0.0 0.1506 revolute active  0.0000
//standard -1.5708  0.0000 0.0 0.0000 revolute active  0.0000
//standard  1.5708  0.0000 0.0 0.0000 revolute active -1.5708
//standard  0.0000  0.0000 0.0 0.0000 revolute active  1.5708
//mod/std   alpha      a  theta   d   rev/pris act/pas offset

std::ostream& operator << (std::ostream& os,const KDL::JntArray& J)
{
  for (unsigned int i = 0; i < J.rows(); i++) {
    os << J.data[i] << " ";
  }
  os << std::endl;
  return os;
}

int main()
{
    // cisstRobot
    std::string robFileName = "/home/zihan/dev/kdl/source/dvmtm.full.rob";
    robManipulator robCisst(robFileName);
//    robCisst.

    // KDL
    //    Frame::DH(a, alpha, d, theta)
    //    RigidBodyInertia(m, com, RotationalInertia(ixx, iyy, izz, ixy, ixz, iyz));
    Chain robKDL;
    RigidBodyInertia inert;
    // 1
    inert = RigidBodyInertia(0.0, KDL::Vector(0, 0, 0), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 1.5708, 0.0, -1.5708), inert));
    // 2
    inert = RigidBodyInertia(0.10, KDL::Vector(-0.15, 0, 0), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.2794, 0.0, 0.0, -1.5708), inert));
    // 3
    inert = RigidBodyInertia(0.03, KDL::Vector(-0.15, 0, 0), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.3645, -1.5708, 0.0, 1.5708), inert));
    // 4
    inert = RigidBodyInertia(0.05, KDL::Vector(0, -0.15, -0.15), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 1.5708, 0.1506, 0.0), inert));
    // 5
    inert = RigidBodyInertia(0.0, KDL::Vector(0, 0.03, -0.03), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, -1.5708, 0.0, 0.0), inert));
    // 6
    inert = RigidBodyInertia(0.0, KDL::Vector(0, -0.02, 0.02), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 1.5708, 0.0, -1.5708), inert));
    // 7
    inert = RigidBodyInertia(0.0, KDL::Vector(0, 0, 0), RotationalInertia(0.1, 0.1, 0.1, 0.0, 0.0, 0.0));
    robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 0.0, 0.0, 1.5708), inert));

    vctDoubleVec q(7);
    vctFrm4x4 tip;
    q.SetAll(0.0);
//    q[0] = 0.0; q[1] = 1.0; q[2] = 1.0; q[3] = 1.0;
//    q[4] = 1.0; q[5] = 1.0; q[6] = 1.0;

    // Forward Kinematics
    tip = robCisst.ForwardKinematics(q);
    std::cout << "tip = " << tip.Translation() << std::endl;

    vctQuatRot3 rotQuat(tip.Rotation(), VCT_NORMALIZE);
    std::cout << rotQuat.X() << " " << rotQuat.Y() << " "
              << rotQuat.Z() << " " << rotQuat.W() << std::endl;

    // KDL
    ChainFkSolverPos_recursive fkSolver = ChainFkSolverPos_recursive(robKDL);
    unsigned int numJnts = robKDL.getNrOfJoints();
    KDL::JntArray jnt_q = JntArray(numJnts);
    KDL::JntArray jnt_qd = JntArray(numJnts);
    KDL::JntArray jnt_qdd = JntArray(numJnts);
    KDL::JntArray jnt_tau = JntArray(numJnts);

    for (unsigned int i = 0; i < numJnts; i++)
    {
      jnt_q(i) = q[i];
      jnt_qd(i) = 0.0;
      jnt_qdd(i) = 0.0;
    }
    Frame tipFrame;
    fkSolver.JntToCart(jnt_q, tipFrame);
    std::cout << std::endl << tipFrame.p << std::endl;
    double x, y, z, w;
    tipFrame.M.GetQuaternion(x, y, z, w);
    std::cout << x << " " << y << " " << z << " " << w << std::endl;


    // Gravity Compenstation
    vctDoubleVec qd(numJnts, 0.0);     // zero vel
    vctDoubleVec qdd(numJnts, 0.0);    // zero accel
    vctDoubleVec tau(numJnts, 0.0);    // output tau
    tau = robCisst.InverseDynamics(q, qd, qdd);
    std::cout << std::endl
              << "tau = " << tau << std::endl;


    // Gravity 9.8 or -9.8 ???
    ChainIdSolver_RNE idSolver(robKDL, KDL::Vector(0.0, 0.0, -9.81));
    KDL::Wrenches wrenches;
    KDL::Wrench wrench_null;
    for (unsigned int i = 0; i < numJnts; i++)
      wrenches.push_back(wrench_null);
    int ret;
    ret = idSolver.CartToJnt(jnt_q, jnt_qd, jnt_qdd, wrenches, jnt_tau);
    if (ret < 0) std::cerr << "Inverse Dynamics WRONG" << std::endl;

    std::cout << "tau = " << jnt_tau << std::endl;
}
