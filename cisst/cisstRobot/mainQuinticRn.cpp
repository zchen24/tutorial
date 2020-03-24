#include <iostream>
#include <fstream>
#include <cisstVector.h>
#include <cisstRobot/robLinearRn.h>
#include <cisstRobot/robQuintic.h>

int main(int argc, char *argv[])
{
    // Example R2
    //  q(0,0) to q(pi, pi)

    // q0 = [0 0]
    vctDoubleVec q0; q0.SetSize(2); q0.SetAll(0.0);
    vctDoubleVec qd0; qd0.SetSize(2); qd0.SetAll(0.0);
    vctDoubleVec qdd0; qdd0.SetSize(2); qdd0.SetAll(0.0);
    // q1 = [pi pi]
    vctDoubleVec q1; q1.SetSize(2); q1.SetAll(cmnPI);
    vctDoubleVec qd1; qd1.SetSize(2); qd1.SetAll(0.0);
    vctDoubleVec qdd1; qdd1.SetSize(2); qdd1.SetAll(0.0);
    // vmax = [0.05, 0.05]
    vctDoubleVec vmax; vmax.SetSize(2); vmax.SetAll(0.05);


    double tstart = 0.0;
    double tstop = 0.0;

    // define robLinearRn function
    robLinearRn Linear(q0,      // start pos
                       q1,      // stop pos
                       vmax,    // vmax
                       tstart); // start time
    // update tstop
    tstop = Linear.StopTime();


    // define robQuintic function
    robQuintic Quintic(tstart,
                       q0, qd0, qdd0,
                       tstop,
                       q1, qd1, qdd1);


    // simulate the time run
    double tstep = 0.1;
    vctDoubleVec q; q.SetSize(2);
    vctDoubleVec qd; qd.SetSize(2);
    vctDoubleVec qdd; qdd.SetSize(2);

    // let's log the data to file
    std::ofstream logLinear;
    logLinear.open("LinearRnLog.txt");
    std::ofstream logQuintic;
    logQuintic.open("QuinticRnLog.txt");
    for (double t = tstart; t < tstop; t = t + tstep)
    {
        Linear.Evaluate(t, q, qd, qdd);
        logLinear << std::fixed << std::setprecision(3)
                  << "t = " << t << "  q = " << q << std::endl;

        Quintic.Evaluate(t, q, qd, qdd);
        logQuintic << std::fixed << std::setprecision(3)
                   << "t = " << t << "  q = " << q << std::endl;
    }
    logLinear.close();
    logQuintic.close();

    return 0;
}
