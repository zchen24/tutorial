cmake_minimum_required(VERSION 3.0)
project(soem_ecat)

find_package(soem REQUIRED)
add_executable(soem_ecat main.cpp)
target_link_libraries(soem_ecat soem)
