cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)

# NOTE: 
# CUDA is 1st class support since 3.1.0

add_executable(dist_v1 mainDistV1.cpp)
add_executable(dist_v1_cuda mainDistV1.cu)

add_executable(dist_v2 mainDistV2.cpp)
add_executable(dist_v2_cuda mainDistV2Cuda.cpp distv2kernel.h distv2kernel.cu)