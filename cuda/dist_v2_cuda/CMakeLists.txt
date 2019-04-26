cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 11)

add_executable(dist_v2_cuda dist_v2_cuda.cu)
