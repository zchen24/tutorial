cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(tricoloring LANGUAGES CXX CUDA)

add_executable(tricoloring tricoloring.cu)
target_link_libraries(tricoloring nppidei nppial nppif)
