cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(boost_tutorial)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED thread)
include_directories(${Boost_INCLUDE_DIRS})

# Set the ouptut path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# string example
add_executable(shared_ptr shared_ptr_test.cpp)

# boost thread
add_executable(b_thread b_thread.cpp)
target_link_libraries(b_thread ${Boost_LIBRARIES})

add_executable(b_thread_interrupt b_thread_interrupt.cpp)
target_link_libraries(b_thread_interrupt ${Boost_LIBRARIES})

add_executable(b_thread_detach b_thread_detach.cpp)
target_link_libraries(b_thread_detach ${Boost_LIBRARIES})

# boost format
add_executable(b_format b_format.cpp)


