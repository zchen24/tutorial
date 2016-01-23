// 2016-01-23
// Zihan Chen

#include <iostream>

int myAdd(int a, int b)
{
    return (a+b);
}


int main(int argc, char *argv[])
{
  std::cout << "Callgrind" << std::endl;

  for (size_t i = 0; i < 10; i++)
  {
      myAdd(1, i);
  }

  return 0;
}
