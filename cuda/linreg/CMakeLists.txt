cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(linreg LANGUAGES CXX CUDA)


add_executable(linreg linreg.cu)
target_link_libraries(linreg cublas cusolver)