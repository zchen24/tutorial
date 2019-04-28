cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(centroid_2d LANGUAGES CXX CUDA)

add_executable(centroid_2d centroid_2d.cu)
