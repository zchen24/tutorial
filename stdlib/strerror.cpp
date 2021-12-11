//! @brief demo how to get error code's string
//! @author Zihan Chen
//! @date 2021-12-10


#include <iostream>
#include <cstring>

int main(int, char**)
{
    std::cout << "Error code: " << EACCES << " " << strerror(EACCES) << "\n";
    return 0;
}
