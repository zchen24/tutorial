cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(sharpen_npp LANGUAGES CXX CUDA)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(sharpen_npp sharpen_npp.cu)
target_link_libraries(sharpen_npp nppif)
