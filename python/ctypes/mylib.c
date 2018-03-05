// On Windows:
// use cmake to compile
//
// On Linux:
// gcc -o mylib.so -shared -fPIC mylib.c

#ifdef _WIN32
#define WINAPI _declspec(dllexport)
#else
#define WINAPI 
#endif

int WINAPI my_add(int x, int y)
{
    return (x + y);
}

void WINAPI get_a_value(int* val)
{
    (*val) = 100;
}
