cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(stdlib_tutorial)
set(CMAKE_CXX_STANDARD 11)

# Set the output path for the libraries and executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

# getopt example
add_executable(getopt_test getopt_test.cpp)
#target_link_libraries(getopt_test raw1394)

# getopt_long example
add_executable(getopt_long_test getopt_long_test.cpp)

# std::string example
add_executable(string_test string_test.cpp)

# std::pair example
add_executable(pair_test pair_test.cpp)

# std::cout example
add_executable(cout_test cout_test.cpp)

# std::lock_guard example
add_executable(lock_guard_test lock_guard_test.cpp)
target_link_libraries(lock_guard_test pthread)

# std::thread example
add_executable(thread_test thread_test.cpp)
target_link_libraries(thread_test pthread)

# std::deque
add_executable(deque_test deque_test.cpp)
target_link_libraries(deque_test)

# poll and fgets
add_executable(key_test key_test.cpp)

# time difference
add_executable(time_diff time_diff.cpp)

# std::atomic
add_executable(std_atomic std_atomic.cpp)
target_link_libraries(std_atomic pthread)

# std::bind
add_executable(std_bind std_bind.cpp)

# std::chrono
add_executable(std_chrono std_chrono.cpp)

# std::function
add_executable(std_function std_function.cpp)

# std::sleep
add_executable(std_sleep std_sleep.cpp)

# std::set
add_executable(std_set std_set.cpp)

# std::tuple
add_executable(std_tuple std_tuple.cpp)

# std::map
add_executable(std_map std_map.cpp)

add_executable(class_override class_override.cpp)
