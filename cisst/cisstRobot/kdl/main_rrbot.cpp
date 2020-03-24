// Compare cisstRobot & KDL using Standard DH
// Zihan Chen
// 2014-07-24

#include <iostream>
#include <cisstRobot.h>
#include <cisstVector.h>
#include <kdl/chain.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainjnttojacsolver.hpp>

using namespace KDL;

//standard  1.5708  0.0000 0.0 0.0000 revolute active -1.5708
//standard  0.0000  0.2794 0.0 0.0000 revolute active -1.5708
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
    std::string robFileName = "/home/zihan/ros/gravity/gazebo_ros_demos/rrbot_description/DH/rrbot.rob";
    robManipulator robCisst(robFileName);

    std::string robFileNameMod = "/home/zihan/ros/gravity/gazebo_ros_demos/rrbot_description/DH/rrbot.mod.rob";
    robManipulator robCisstMod(robFileNameMod);

    int numJnts = robCisst.links.size();
    vctDoubleVec q(numJnts, 0.0);
    vctDoubleVec qd(numJnts, 0.0);     // zero vel
    vctDoubleVec qdd(numJnts, 0.0);    // zero accel
    vctDoubleVec tau(numJnts, 0.0);    // output tau

    int numJntsMod = robCisstMod.links.size();
    vctDoubleVec qm(numJntsMod, 0.0);
    vctDoubleVec qdm(numJntsMod, 0.0);     // zero vel
    vctDoubleVec qddm(numJntsMod, 0.0);    // zero accel
    vctDoubleVec taum(numJntsMod, 0.0);    // output tau

    q[0] = qm[0] = 1.5708;

    // Forward Kinematics
    vctFrm4x4 tip;
    tip = robCisst.ForwardKinematics(q);
    std::cout << "tip Std = " << tip.Translation() << std::endl;

    vctQuatRot3 rotQuat(tip.Rotation(), VCT_NORMALIZE);
    std::cout << rotQuat.X() << " " << rotQuat.Y() << " "
              << rotQuat.Z() << " " << rotQuat.W() << std::endl;

    vctFrm4x4 tipMod;
    tipMod = robCisstMod.ForwardKinematics(qm);
    std::cout << "tip Mod = " << tipMod.Translation() << std::endl;
    vctQuatRot3 rotQuatMod(tipMod.Rotation(), VCT_NORMALIZE);
    std::cout << rotQuatMod.X() << " " << rotQuatMod.Y() << " "
              << rotQuatMod.Z() << " " << rotQuatMod.W() << std::endl;

#if 1
    // Rotate the base
    vctMatrixRotation3<double> Rw0(  0.0,  0.0, -1.0,
                                     0.0,  1.0,  0.0,
                                     1.0,  0.0,  0.0 );
    vctFixedSizeVector<double,3> tw0(0.0);
    vctFrame4x4<double> Rtw0( Rw0, tw0 );
    robCisst.Rtw0 = Rtw0;
    robCisstMod.Rtw0 = Rtw0;

    // Gravity Compenstation
    tau = robCisst.InverseDynamics(q, qd, qdd);
    std::cout << std::endl
              << "tau std = " << tau << std::endl;

    taum.SetAll(0.0);
    taum = robCisstMod.InverseDynamics(qm, qdm, qddm);
    std::cout << "tau mod = " << taum << std::endl;
#endif
}
