cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)

project(opencv_cuda)

# ===========================
# Cuda
# ===========================
add_executable(cvCudaResize CudaResize.cpp)
target_link_libraries(cvCudaResize ${OpenCV_LIBRARIES})

add_executable(cvCuda101 Cuda101.cpp)
target_link_libraries(cvCuda101 ${OpenCV_LIBRARIES})
