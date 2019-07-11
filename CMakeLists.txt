cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(tuturial)

option(BUILD_STDLIB "Build stdlib examples" ON)
if(BUILD_STDLIB)
    add_subdirectory(stdlib)
endif(BUILD_STDLIB)

option(BUILD_BOOST "Build boost examples" OFF)
if(BUILD_BOOST)
    add_subdirectory(boost)
endif(BUILD_BOOST)

option(BUILD_EIGEN "Build Eigen examples" OFF)
if(BUILD_EIGEN)
    add_subdirectory(eigen)
endif(BUILD_EIGEN)

option(BUILD_OPENCV "Build OpenCV examples" OFF)
if(BUILD_OPENCV)
    add_subdirectory(opencv)
endif(BUILD_OPENCV)

#add_subdirectory(raw1394)
#add_subdirectory(json)
#add_subdirectory(kdl)


