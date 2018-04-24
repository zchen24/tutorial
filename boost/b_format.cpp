#include <iostream>
#include <boost/format.hpp>

int main(int argc, char** argv)
{
    boost::format format("One string: %s and one double %f");
    format % "hello" % 3.14;
    std::cout << format.str() << "\n";
}