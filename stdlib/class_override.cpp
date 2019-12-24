#include <iostream>


class Base {
public:
    Base() { std::cout << "Base Constructor\n";}
    virtual ~Base() { std::cout << "Base Destructor\n";}
};

class Child : public Base {
public:
    Child() { std::cout << "Child Constructor\n";}
    ~Child() override { std::cout << "Child Destructor\n"; }
};

class GrandChild : public Child {
public:
    GrandChild() { std::cout << "GrandChild Constructor\n"; }

    ~GrandChild() override { std::cout << "GrandChild Destructor\n"; }
};

int main() { GrandChild gc; }