cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(opencv_tutorial)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cvKernel src/Kernel.cpp)
target_link_libraries(cvKernel ${OpenCV_LIBRARIES})

add_executable(cvBlending src/Blending.cpp)
target_link_libraries(cvBlending ${OpenCV_LIBRARIES})

add_executable(cvContrast src/Contrast.cpp)
target_link_libraries(cvContrast ${OpenCV_LIBRARIES})

add_executable(cvDrawing src/Drawing.cpp)
target_link_libraries(cvDrawing ${OpenCV_LIBRARIES})

# file(COPY imgs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_CURRENT_SOURCE_DIR}/imgs
                ${CMAKE_CURRENT_BINARY_DIR})
