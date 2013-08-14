cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(protobuf_tutorial)

# protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# generate protobuf files
include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS msg/person.proto)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS msg/addressbook.proto)

# output example
add_executable(output src/output.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(output ${PROTOBUF_LIBRARIES})

# input example
add_executable(input src/input.cpp ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(input ${PROTOBUF_LIBRARIES})
