// use cmake to compile

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