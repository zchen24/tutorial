#include <iostream>
#include <string.h>


//! @brief Sample code showing string operation
//! @author Zihan Chen
//! @date 2015-12-12


int main(int argc, char** argv)
{
  std::string str = "This_is_a_test_string";

  // get substring
  std::string substr = str.substr(0, 4);
  std::cout << "sub str = " << substr << std::endl;

  // find 1st substring
  int pos = str.find("is");
  std::cout << "1st 'is' is at = " << pos << std::endl;

  // check if has substring
  const char* t1 = strstr(str.c_str(), "test");
  std::cout << "t1 string = " << t1 << std::endl;

  const char* t2 = strstr(str.c_str(), "bullshit");
  if (t2 == NULL) {
    std::cout << "t2 is NULL" << std::endl;
  } else {
    std::cout << "t2 string = " << t2 << std::endl;
  }
}
