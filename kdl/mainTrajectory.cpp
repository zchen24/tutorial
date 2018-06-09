// License BSD
// Zihan Chen
// 2018-06-08
// Shows how to use trajectory in KDL

#include <iostream>
#include <kdl/path_line.hpp>
#include <kdl/rotational_interpolation_sa.hpp>
#include <kdl/velocityprofile_trap.hpp>
#include <kdl/trajectory_segment.hpp>
#include <kdl/trajectory_stationary.hpp>
#include <kdl/trajectory_composite.hpp>
#include <kdl/frames_io.hpp>


using namespace KDL;

int main(int argc, char** argv)
{
    Frame t_start(Vector(0, 0, 0));
    Frame t_end(Vector(1, 1, 1));
    double max_vel = 4.0;
    double max_acc = 100.0;

    double eqradius = 1.0;
    auto path = new Path_Line(t_start,
                              t_end,
                              new RotationalInterpolation_SingleAxis(),
                              eqradius);
    VelocityProfile* vel_profile = new VelocityProfile_Trap(max_vel, max_acc);
    Trajectory* traj_1 = new Trajectory_Segment(path, vel_profile, 1.0);
    Trajectory* traj_2 = new Trajectory_Stationary(1.0, t_end);
    Trajectory_Composite trajectory;
    trajectory.Add(traj_1);
    trajectory.Add(traj_2);

    std::cout << "Duration = " << trajectory.Duration() << "\n";

    double dt = 0.01;
    // trajectory outputs the end pose when t is over Duration
    for (double t = 0; t < trajectory.Duration()+0.5; t+=dt) {
        Frame now = trajectory.Pos(t);
        std::cout << "t = " << t << "    pos = " << now.p << '\n';
    }
}


