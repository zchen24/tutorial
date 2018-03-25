
#include <iostream>
#include "MyClass.h"

MyClass::MyClass():
    m_int(10)
{}

void MyClass::PrintMyInt()
{
    std::cout << "My Int = " << m_int;
}


