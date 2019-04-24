cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)


add_executable(dist_v1_cuda ../cuda_utils.h dist_v1.cu)
