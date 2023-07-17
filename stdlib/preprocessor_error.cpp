//! @brief Demo how to throw an #error in preprocessor
//! @author Zihan Chen
//! @date 2023-07-17

#include <iostream>

#ifdef __WIN32__
#include <Windows.h>
#else
#error "#include failed, not a Windows system"
#endif

int main(int, char**)
{
    printf("Test #include error");
}