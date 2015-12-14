#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
  std::cout << "optimized DFT size = " << cv::getOptimalDFTSize(11) << std::endl;

  std::cout << "Kernel" << std::endl;
  return 0;
}
