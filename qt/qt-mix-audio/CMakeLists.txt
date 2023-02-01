cmake_minimum_required(VERSION 3.0)
project(qt_mix_audio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets Multimedia REQUIRED)

add_executable(qt_mix_audio qt-mix-audio.cpp)
target_link_libraries(qt_mix_audio Qt5::Core Qt5::Widgets Qt5::Multimedia)

add_custom_command(
  TARGET qt_mix_audio
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/../assets/example.wav
          ${CMAKE_CURRENT_BINARY_DIR}/example.wav
)