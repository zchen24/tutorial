cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(curand_pi LANGUAGES CXX CUDA)

add_executable(curand_pi curand_pi.cu)
target_link_libraries(curand_pi curand)
