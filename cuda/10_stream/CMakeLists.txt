cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(stream LANGUAGES CXX CUDA)

add_executable(pinned_mem pinned_mem.cu)
add_executable(single_stream single_stream.cu)
add_executable(multi_stream multi_stream.cu)
