// Examples for Eigen library for linear algebra
//
// Date: 2018-12-11


#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
using Eigen::MatrixXd;

using namespace std;

int main()
{

    cout << "fmod(2*pi,2*pi) = " << fmod(0.0 + 2 * M_PI, 2*M_PI) << "\n";

    // -------------------------
    // Vector
    // -------------------------
    cout << "----------------------------------\n"
         << "Vector\n"
         << "----------------------------------\n\n";

    // fixed length
    Eigen::Vector3d v1(1,2,3);
    Eigen::Vector3d v2;
    v2 << 4, 5, 6;
    cout << "v1 = " << v1.transpose() << "\n"
         << "v2 = " << v2.transpose() << "\n";

    // indexing
    v1(1) = 10;
    cout << "v1(0) = " << v1(0) << "\n"
         << "v1(1) = " << v1(1) << "\n";

    // variable
    Eigen::VectorXd vv(7);
    vv << 7,8,9,10,11,12,13;
    cout << "VectorXd vv = " << vv.transpose() << "\n";
    cout << "vv.size() = " << vv.size() << "\n";


    // element access
    MatrixXd m(2,2);
    MatrixXd m22(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    m22 = m;
    std::cout << "initial 2x2 matrix\n"
              << m << std::endl;

    // resize
    m.resize(3,3);
    m << 1,2,3,
         4,5,6,
         7,8,9;
    std::cout << "resized to (3,3)\n"
              << m << std::endl;


    // assignment
    m.block<2,2>(1,1) = m22.block<2,2>(0,0);
    std::cout << "m2 = \n" << m22 << "\n"
              << "m block access = \n" << m << "\n";


    // -------------------------
    // Matrix slicing & blocks
    // -------------------------

//    m.block<2,2>(0,0);
//    m.topRows<2>();

    // Least square

    // -------------------------
    // SVD
    // -------------------------
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullU|Eigen::ComputeFullV);
    auto U = svd.matrixU();
    auto S = svd.singularValues();
    auto V = svd.matrixV();

    Eigen::MatrixXd m_from_svd = U * S.asDiagonal() * V.transpose();

    std::cout << "\n\n"
              << "U = \n" << svd.matrixU()
              << "\nS = \n" << svd.singularValues()
              << "\nV = \n" << svd.matrixV()
              << "\n\n m = \n" << m << "\n"
              << "\n\n m_from_svd = U * S * V'\n"
              << m_from_svd
              << "\n";

    // -------------------------
    // Pseudo Inverse
    // -------------------------
    std::cout << "S.size() = " << S.size() << "\n"
              << "S(0) = " << S(0) << "  " << S(1) << "  " << S(2) << "\n";

    S(0) = 1.0 / S(0);
    S(1) = 1.0 / S(1);
    S(2) = 1.0 / S(2);
    Eigen::MatrixXd m_inv = V * S.asDiagonal() * U.transpose();
    std::cout << "\n\nm_inv = \n" << m_inv << "\n"
              << "\n m * m_inv = \n" << m * m_inv << "\n";
}
