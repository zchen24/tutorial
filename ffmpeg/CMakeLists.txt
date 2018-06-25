cmake_minimum_required(VERSION 2.8.3)
project(ffmpeg_tutorial)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

find_library(AVCODEC_LIB avcodec)
find_library(AVUTIL_LIB avutil)
find_library(AVFORMAT_LIB avformat)
find_library(SWSCALE_LIB swscale)

set(ALL_TUTORIALS tutorial_01 tutorial_02 tutorial_03)
foreach(tutorial ${ALL_TUTORIALS})
    add_executable(${tutorial} ${tutorial}.cpp)
    target_link_libraries(${tutorial}
            ${AVCODEC_LIB} ${AVUTIL_LIB} ${AVFORMAT_LIB} ${SWSCALE_LIB}
            ${OpenCV_LIBS} ${SDL2_LIBRARIES})
endforeach()
