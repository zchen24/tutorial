#include <iostream>
#include <cisstVector.h>

int main(int argc, char *argv[])
{
    std::cout << "hello cisstVector" << std::endl;

    vct3 v1(1.0, 1.0, 1.0);
    vct3 v2(1.0, 1.0, 1.0);
    std::cout << v1 << std::endl;

    v1.NormalizedSelf();
    std::cout << v1 << std::endl;

    std::cout << v1.NormSquare() << std::endl;

    v1.Subtract(v2);
    std::cout << v1 << std::endl;

    v1.Multiply(2);
    std::cout << v1 << std::endl;
    v1.Divide(2);
    std::cout << v1 << std::endl;

    std::cout << v1.DotProduct(v2) << std::endl;
    return 0;
}

