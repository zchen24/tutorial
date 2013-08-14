// Example code from Google protobuf tutorial

#include <iostream>
#include <fstream>
#include "person.pb.h"

int main(int argc, char** argv)
{
    // generate msg
    Person person;
    person.set_name("John Doe");
    person.set_id(1234);
    person.set_email("jdoe@example.com");

    // output
    std::fstream output("myfile", std::ios::out | std::ios::binary);
    person.SerializeToOstream(&output);
}
