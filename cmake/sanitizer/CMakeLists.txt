cmake_minimum_required(VERSION 3.17)

project(CSROS LANGUAGES CXX C)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")

add_executable(San main.cpp)
