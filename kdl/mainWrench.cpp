// Zihan Chen
// 2014-09-05

// Demos how to transform wrench
// w:   wrench measured using force sensor (in FT Sensor Frame)
// T:   transform from FT sensor frame to EE (End Effector) frame
// wee: wrench expressed in EE frame
// wee = T * w
//
// Matlab
// J = tr2jac(T)
// wee = J' * w
//
// Reference: RVC Book 2013 P186


#include <kdl/frames.hpp>
#include <kdl/frames_io.hpp>
#include <iostream>

int main()
{
    using namespace KDL;

    Wrench w(Vector(1,0,0),
             Vector(0,0,0));
    Wrench wee;
    Frame T(Rotation::RotY(-0.3142), Vector(0, 0, 0.15));
    wee = T * w;

    std::cout << "T = " << T << std::endl << std::endl;

    std::cout << "w = " << w << std::endl
              << "wt = " << wee << std::endl;

    std::cout << wee[0] << " " << wee[1] << " " << wee[2] << std::endl;
}
