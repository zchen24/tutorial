// Example code from Google protobuf tutorial

#include <iostream>
#include <fstream>
#include "person.pb.h"

int main(int argc, char** argv)
{
    std::fstream input("myfile", std::ios::in | std::ios::binary);
    Person person;
    person.ParseFromIstream(&input);

    std::cout << "Name: " << person.name() << std::endl;
    std::cout << "Email: " << person.email() << std::endl;
}
