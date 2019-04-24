cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(norm LANGUAGES CXX CUDA)

add_executable(norm kernel.cu)
