cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(tuturial)

option(BUILD_FLIR "Build FLIR examples" OFF)
if(BUILD_FLIR)
    add_subdirectory(FLIR)
endif(BUILD_FLIR)

option(BUILD_STDLIB "Build stdlib examples" ON)
if(BUILD_STDLIB)
    add_subdirectory(stdlib)
endif(BUILD_STDLIB)

option(BUILD_BOOST "Build boost examples" OFF)
if(BUILD_BOOST)
    add_subdirectory(boost)
endif(BUILD_BOOST)

option(BUILD_CISST "Build cisst examples" OFF)
if(BUILD_CISST)
    add_subdirectory(cisst)
endif(BUILD_CISST)

option(BUILD_EIGEN "Build Eigen examples" OFF)
if(BUILD_EIGEN)
    add_subdirectory(eigen)
endif(BUILD_EIGEN)

option(BUILD_OPENCV "Build OpenCV examples" OFF)
if(BUILD_OPENCV)
    add_subdirectory(opencv)
endif(BUILD_OPENCV)

option(BUILD_PYBIND11 "Build Pybind11 examples" OFF)
if(BUILD_PYBIND11)
    add_subdirectory(pybind11)
endif(BUILD_PYBIND11)

option(BUILD_SOEM "Build SOEM examples" OFF)
if(BUILD_SOEM)
    add_subdirectory(soem)
endif(BUILD_SOEM)

#add_subdirectory(raw1394)
#add_subdirectory(json)
#add_subdirectory(kdl)

option(BUILD_QT "Build Qt examples" OFF)
if(BUILD_QT)
    add_subdirectory(qt)
endif(BUILD_QT)

