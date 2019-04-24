cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)

add_executable(dist_v2_cuda_unified dist_v2_unified.cu)
