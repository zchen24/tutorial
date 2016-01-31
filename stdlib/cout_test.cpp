#include <iostream>
#include <iomanip>    // setprecision


//! @brief Sample code showing std::cout
//! @author Zihan Chen
//! @date 2016-01-31
//! @ref http://en.cppreference.com/w/cpp/io/manip/setprecision


int main(int argc, char** argv)
{
  double number = 10003.1415916;

  // precision after dot
  std::cout << std::fixed << std::setprecision(3) << number << std::endl;
}
