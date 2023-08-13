cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(posix_tutorial)
set(CMAKE_CXX_STANDARD 11)

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# std::thread example
add_executable(pthread_stack pthread_stack.cpp)
target_link_libraries(pthread_stack pthread)


# sched_fifo_priority
add_executable(sched_fifo_priority sched_fifo_priority.cpp)

# sched_fifo_two_threads
add_executable(sched_fifo_two_threads sched_fifo_two_threads.cpp)
target_link_libraries(sched_fifo_two_threads pthread)
