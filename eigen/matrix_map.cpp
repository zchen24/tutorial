// Eigen Map
//
// Date: 2019-06-20


#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main()
{
    int array[9];
    for (int i = 0; i < 9; ++i) array[i] = i;
    cout << "array original: \n";
    for (auto a : array) { cout << a << "  "; }
    cout << "\n\n";

    Eigen::Map<Eigen::Matrix3i> em(array);
    cout << "em: \n" << em << "\n\n";

    // changing elements
    em(0, 0) = 20;
    cout << "After em(0,0) = 20:\n" << em << "\n\n";
    cout << "array changed: \n";
    for (auto a : array) { cout << a << "  "; }
    cout << "\n\n";

    // copied matrix
    Eigen::Matrix3i cm = Eigen::Map<Eigen::Matrix3i>(array);
    cout << "cm: \n" << cm << "\n\n";
    cm(0, 0) = 10;
    cout << "After cm(0,0) = 10:\n" << cm << "\n\n";
    cout << "array not changed: \n";
    for (auto a : array) { cout << a << "  "; }
    cout << "\n\n";

    // ---------------
    // dynamic map
    // ---------------
    Eigen::Map<Eigen::VectorXi> mvxi(array, 5);
    cout << "Map<VectorXi>.transpose() = " << mvxi.transpose() << "\n\n";
}
