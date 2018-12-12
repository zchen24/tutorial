#include <iostream>
#include <iomanip>    // setprecision


//! @brief Sample code showing std::cout
//! @author Zihan Chen
//! @date 2016-01-31
//! @ref http://en.cppreference.com/w/cpp/io/manip/setprecision


int main(int argc, char** argv)
{
    double number = 10003.1415916;
    std::cout << number << std::endl;

    // save cout format
    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    // precision after dot
    std::cout << std::fixed << std::setprecision(3) << number << std::endl;
    std::cout << number << std::endl;

    // restore cout format
    std::cout.copyfmt(oldState);
    std::cout << number << std::endl;
}
