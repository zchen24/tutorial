cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(stdlib_tutorial)

# Set the ouptut path for the libraries and executables
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

# poll and fgets
add_executable(key_test key_test.cpp)
set_target_properties(key_test PROPERTIES
    CXX_STANDARD 11)
