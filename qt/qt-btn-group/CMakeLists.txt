cmake_minimum_required(VERSION 2.8.11)
project(qt_btn_group)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)

add_executable(qt_btn_group qt-btn-group.cpp)
qt5_use_modules(qt_btn_group Widgets)
