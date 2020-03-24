// Zihan Chen
// 2014-07-24
// Compare KDL standard & modified DH
//  - cisstRobot modified DH
//  - KDL standard DH
//  - compare FK & Jac
//
// Conclusion
//  - cisstRobot Modified DH FK & Jac correct
//  - KDL standard DH correct


#include <iostream>
#include <cisstRobot.h>
#include <cisstVector.h>
#include <cisstCommon/cmnPortability.h>
#include <kdl/chain.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainjnttojacsolver.hpp>


using namespace KDL;


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
#if 1
    // KDL
    //    Frame::DH(a, alpha, d, theta)
    //    RigidBodyInertia(m, com, RotationalInertia(ixx, iyy, izz, ixy, ixz, iyz));
     Chain robKDL;
     // 0
     robKDL.addSegment(Segment(Joint(Joint::None), Frame::DH(0.0, 1.5708, 0.0, 0)));
     // 1
     robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, -1.5708, 0.0, 1.5708)));
     // 2    
     robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 1.5708, 0.0, -1.5708)));
     // 3
     robKDL.addSegment(Segment(Joint(Joint::TransZ), Frame::DH(0.0, 0.0, -(0.4318-0.4162), 0.00)));
     // 4
     robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, -1.5708, 0.0, 0.0)));
     // 5
     robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0091, -1.5708, 0.0, -1.5708)));
     // 6
     robKDL.addSegment(Segment(Joint(Joint::RotZ), Frame::DH(0.0, 0.0, 0.0, -1.5708)));


    // KDL
    ChainFkSolverPos_recursive fkSolver = ChainFkSolverPos_recursive(robKDL);
    unsigned int numJnts = robKDL.getNrOfJoints();
    KDL::JntArray jnt_q = JntArray(numJnts);
    KDL::JntArray jnt_qd = JntArray(numJnts);
    KDL::JntArray jnt_qdd = JntArray(numJnts);

    std::cout << "KDL nj = " << numJnts << std::endl;

    for (unsigned int i = 0; i < numJnts; i++)
    {
      jnt_q(i) = 0.0;
      jnt_qd(i) = 0.0;
      jnt_qdd(i) = 0.0;
    }
    Frame tipFrame;
    fkSolver.JntToCart(jnt_q, tipFrame);    
    double x, y, z, w;
    tipFrame.M.GetQuaternion(x, y, z, w);

    std::cout << "----- KDL fk ---------" << std::endl
              << tipFrame.p << std::endl
              << x << " " << y << " " << z << " " << w << std::endl;

    // KDL
    ChainJntToJacSolver jacSolver(robKDL);
    KDL::Jacobian jac_kdl(numJnts);
    KDL::Jacobian jac_body_kdl(numJnts);
    jacSolver.JntToJac(jnt_q, jac_kdl);

    // R0 w.r.t. R6
    // -1  0  0
    //  0  0 -1
    //  0 -1  0
    KDL::changeBase(jac_kdl, KDL::Rotation(-1, 0, 0, 0, 0, -1, 0, -1, 0), jac_body_kdl);
    std::cout << std::endl
              << "Jacobian Body KDL" << std::endl
              << jac_body_kdl.data << std::endl;
#endif

    // cisstRobot
    std::string robFileName = "/home/zihan/dev/cisst/source/sawIntuitiveResearchKit/share/dvpsm.rob";
    robManipulator robCisst(robFileName);

    int numJntsRob = robCisst.links.size();
    vctDoubleVec q(numJntsRob, 0.0);
    vctDoubleVec qd(numJntsRob, 0.0);     // zero vel
    vctDoubleVec qdd(numJntsRob, 0.0);    // zero accel
    vctDoubleVec tau(numJntsRob, 0.0);    // output tau
    vctFrm4x4 tip;

    // Forward Kinematics
    tip = robCisst.ForwardKinematics(q);
    vctQuatRot3 rotQuatMod(tip.Rotation(), VCT_NORMALIZE);
    std::cout << std::endl << "----- cisst ------ " << std::endl
              << tip.Translation() << std::endl
              << rotQuatMod.X() << " " << rotQuatMod.Y() << " "
              << rotQuatMod.Z() << " " << rotQuatMod.W() << std::endl;


#if 1
    // Jacobian
    vctDynamicMatrix<double> J( 6, robCisst.links.size(), VCT_COL_MAJOR );
    robCisst.JacobianBody(q);
    for( size_t r=0; r<6; r++ ){
        for( size_t c=0; c<robCisst.links.size(); c++ ){
            J[r][c] = robCisst.Jn[c][r];
        }
    }
    std::cout << std::endl
              << "Jacobian Body cisst " << std::endl
              << J << std::endl << std::endl;
#endif

}
