cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(parallel_dot LANGUAGES CXX CUDA)

add_executable(parallel_dot parallel_dot.cu)
