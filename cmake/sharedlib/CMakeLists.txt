cmake_minimum_required(VERSION 2.8)
project(sharedlib)

option(BUILD_SHARED_LIBS "Build shared library when ON" OFF)

add_library(mylib mylib.h mylib.cpp)
add_executable(mylib_test main.cpp)
target_link_libraries(mylib_test mylib)


