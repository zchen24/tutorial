#include <iostream>
#include <string.h>


//! @brief Sample code showing std::pair
//! @author Zihan Chen
//! @date 2016-01-31
//! @ref http://en.cppreference.com/w/cpp/utility/pair


int main(int argc, char** argv)
{
  // basic usage
  std::pair<int, std::string> a_pair;
  a_pair.first = 1;
  a_pair.second = "a_pair";
  std::cout << "a_pair.first  = " << a_pair.first << "   "
            << "a_pair.seconc = " << a_pair.second << "\n";

  // assign value using constructor
  std::pair<int, std::string> b_pair(2, "b_pair");
  std::cout << "b_pair.first  = " << b_pair.first << "   "
            << "b_pair.seconc = " << b_pair.second << "\n";


  // typedef pair type
  typedef std::pair<int, std::string> MyPairType;
  MyPairType c_pair(3, "c_pair");
  std::cout << "c_pair.first  = " << c_pair.first << "   "
            << "c_pair.seconc = " << c_pair.second << "\n";

}
