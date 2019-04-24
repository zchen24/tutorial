cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(dist LANGUAGES CXX CUDA)

add_executable(dist_v2_cuda main.cpp dist_v2.h dist_v2.cu)
