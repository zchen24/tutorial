cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(sharpen_npp LANGUAGES CXX CUDA)

add_executable(sharpen_npp sharpen_npp.cu CImg.h)
target_link_libraries(sharpen_npp X11 nppif)
