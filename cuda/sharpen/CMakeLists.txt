cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)

add_executable(sharpen sharpen.h sharpen.cu main.cpp)
