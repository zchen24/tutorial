cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(pinned_mem LANGUAGES CXX CUDA)

add_executable(pinned_mem pinned_mem.cu)
