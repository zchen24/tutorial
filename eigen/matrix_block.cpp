// Examples for Eigen library for linear algebra
//
// Date: 2018-12-11


#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace std;

int main()
{
    // -------------------------
    // Matrix slicing & blocks
    // -------------------------

    Eigen::MatrixXf m(4, 4);
    m << 1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16;
    std::cout << "m = \n" << m << "\n\n";

    // Block of size(p, q), starting at(i, j)
    // matrix.block(i,j,p,q);
    // matrix.block<p,q>(i,j);        
    std::cout << "m.block<2,2>(0,0)\n" << m.block<2, 2>(0, 0) << "\n\n";

    std::cout << "m.block(0,0,2,2)\n" << m.block(0, 0, 2, 2) << "\n\n";

    // Rows 
    std::cout << "m.row(1): " << m.row(1) << "\n\n";

    // Cols
    std::cout << "m.col(1): \n" << m.col(1) << "\n\n";
}
