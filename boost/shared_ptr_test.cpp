#include <iostream>
#include <boost/shared_ptr.hpp>

int main()
{
  std::cout << "shared_ptr" << std::endl;

  // init string
  boost::shared_ptr<std::string> strptr(new std::string("test string"));
  std::cout << "string = " << *(strptr.get()) << std::endl;

  // get another pointer
  std::cout << "count  = " << strptr.use_count() << std::endl;
  boost::shared_ptr<std::string> strptr2 = strptr;
  std::cout << "count  = " << strptr.use_count() << std::endl;

  // reset
  strptr.reset(new std::string("new string"));
  std::cout << "strptr  count  = " << strptr.use_count() << std::endl;
  std::cout << "strptr2 count  = " << strptr2.use_count() << std::endl;
  std::cout << "string = " << *(strptr.get()) << std::endl;
  std::cout << "string = " << *(strptr2.get()) << std::endl;
}
