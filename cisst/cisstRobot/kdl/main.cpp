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
    std::string robFileName = "/home/zihan/Dropbox/dev/cisst_example/source/cisstRobot/kdl/dvmtm.rob";
    robManipulator robCisst(robFileName);

    std::string robFileNameMod = "/home/zihan/Dropbox/dev/cisst_example/source/cisstRobot/kdl/dvmtm.mod.rob";
    robManipulator robCisstMod(robFileNameMod);

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

    unsigned int numJnts = robKDL.getNrOfJoints();

    vctDoubleVec q(numJnts, 0.0);
    vctDoubleVec qd(numJnts, 0.0);     // zero vel
    vctDoubleVec qdd(numJnts, 0.0);    // zero accel
    vctDoubleVec tau(numJnts, 0.0);    // output tau
    //    q[0] = 0.0; q[1] = 1.0; q[2] = 1.0; q[3] = 1.0;
    //    q[4] = 1.0; q[5] = 1.0; q[6] = 1.0;

    // Forward Kinematics
    vctFrm4x4 tip;
    tip = robCisst.ForwardKinematics(q);
    std::cout << "tip Std = " << tip.Translation() << std::endl;

    vctQuatRot3 rotQuat(tip.Rotation(), VCT_NORMALIZE);
    std::cout << rotQuat.X() << " " << rotQuat.Y() << " "
              << rotQuat.Z() << " " << rotQuat.W() << std::endl;

    vctFrm4x4 tipMod;
    tipMod = robCisstMod.ForwardKinematics(q);
    std::cout << "tip Mod = " << tipMod.Translation() << std::endl;
    vctQuatRot3 rotQuatMod(tipMod.Rotation(), VCT_NORMALIZE);
    std::cout << rotQuatMod.X() << " " << rotQuatMod.Y() << " "
              << rotQuatMod.Z() << " " << rotQuatMod.W() << std::endl;


    // KDL
    ChainFkSolverPos_recursive fkSolver = ChainFkSolverPos_recursive(robKDL);    
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


#if 1
    // Gravity Compenstation
    tau = robCisst.InverseDynamics(q, qd, qdd);
    std::cout << std::endl
              << "tau std = " << tau << std::endl;

    tau.SetAll(0.0);
    tau = robCisstMod.InverseDynamics(q, qd, qdd);
    std::cout << "tau mod = " << tau << std::endl;
#endif


    // Gravity ([0 0 -9.81] is gravity vector, w.r.t frame 0)
    ChainIdSolver_RNE idSolver(robKDL, KDL::Vector(0.0, 0.0, -9.81));
    KDL::Wrenches wrenches;
    KDL::Wrench wrench_null;
    for (unsigned int i = 0; i < numJnts; i++)
      wrenches.push_back(wrench_null);
    int ret;
    ret = idSolver.CartToJnt(jnt_q, jnt_qd, jnt_qdd, wrenches, jnt_tau);
    if (ret < 0) std::cerr << "Inverse Dynamics WRONG" << std::endl;

    std::cout << "tau kdl = " << jnt_tau << std::endl;


#if 1
    // ---------- Jacobian Test ----------------
    // Convention used
    // Spatial Jacobian: Ref Frame = Base Frame   Ref Point = Base (F0)
    // Body Jacobian: Ref Frame = Tip Frame   Ref Poit = Tip

    // ZC: the spatial jacobian in cisstRobot ref point is base
    robCisst.JacobianSpatial(q);
    vctDynamicMatrix<double> J( 6, robCisst.links.size(), VCT_COL_MAJOR );
    for( size_t r=0; r<6; r++ ){
        for( size_t c=0; c<robCisst.links.size(); c++ ){
            J[r][c] = robCisst.Js[c][r];
        }
    }
    std::cout << "Jacobian Spatial cisst " << std::endl
              << J << std::endl << std::endl;

    robCisst.JacobianBody(q);
    for( size_t r=0; r<6; r++ ){
        for( size_t c=0; c<robCisst.links.size(); c++ ){
            J[r][c] = robCisst.Jn[c][r];
        }
    }
    std::cout << "Jacobian Body cisst " << std::endl
              << J << std::endl << std::endl;


    robCisstMod.JacobianBody(q);
    vctDynamicMatrix<double> JMod( 6, robCisstMod.links.size(), VCT_COL_MAJOR );
    for( size_t r=0; r<6; r++ ){
        for( size_t c=0; c<robCisstMod.links.size(); c++ ){
            JMod[r][c] = robCisstMod.Jn[c][r];
        }
    }
    std::cout << "Jacobian Body Mod cisst " << std::endl
              << JMod << std::endl << std::endl;

#endif

#if 0
    // WARNING:
    //   - jac_kdl needs number of cols
    //   - KDL::changeBase Rotation is current Base Frame w.r.t. Target Frame
    //   - KDL::changeRefPoint Vector is (PrefNew - PrefOld) w.r.t. current Ref Frame
    ChainJntToJacSolver jacSolver(robKDL);
    KDL::Jacobian jac_kdl(numJnts);
    KDL::Jacobian jac_body_kdl(numJnts);
    KDL::Jacobian jac_spatial_kdl(numJnts);
    ret = jacSolver.JntToJac(jnt_q, jac_kdl);

    KDL::changeRefPoint(jac_kdl, KDL::Vector(-tipFrame.p), jac_spatial_kdl);
    KDL::changeBase(jac_kdl, KDL::Rotation(0, 0, 1, 1, 0, 0, 0, 1, 0), jac_body_kdl);

    std::cout << "ret = " << ret << " row = " << jac_kdl.rows() << " col = " << jac_kdl.columns() << std::endl;
    std::cout << "Jacobian Spatial KDL" << std::endl
              << jac_spatial_kdl.data << std::endl << std::endl;

    std::cout << "Jacobian Body KDL" << std::endl
              << jac_body_kdl.data << std::endl << std::endl;


//    vctRot3 R(-1, 0, 0,
//              0, 0, -1,
//              0, -1, 0);
////    std::cout << R << std::endl;
//    vctDoubleMat T(6, 6);
//    T.SetAll(0.0);

//    T[0][0] = 0;     T[0][1] = 0;     T[0][2] = 1;
//    T[1][0] = 1;     T[1][1] = 0;     T[1][2] = 0;
//    T[2][0] = 0;     T[2][1] = 1;     T[2][2] = 0;

//    T[3][3] = 0;    T[3][4] = 0;     T[3][5] = 1;
//    T[4][3] = 1;     T[4][4] = 0;     T[4][5] = 0;
//    T[5][3] = 0;     T[5][4] = 1;     T[5][5] = 0;

    KDL::Jacobian jac_cb(numJnts);
    for (int i = 0; i < 6; i++) {
      for (int k = 0; k < numJnts; k++) {
        jac_cb.data(i, k) = J[i][k];
      }
    }

    // 0 1 0
    // 0 0 1
    // 1 0 0

    KDL::changeBase(jac_cb, KDL::Rotation(0, 1, 0, 0, 0, 1, 1, 0, 0), jac_cb);
    std::cout << jac_cb.data << std::endl << std::endl;

    std::cout << jac_kdl.data << std::endl;

#endif

//    std::cout << jac_kdl.data << std::endl;
}
