cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(print LANGUAGES CXX CUDA)

add_executable(print printf.cu)
