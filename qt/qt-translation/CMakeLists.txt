cmake_minimum_required(VERSION 3.5)
project(qt_translation)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5
  COMPONENTS Widgets Core LinguistTools REQUIRED)

set(TS_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/lang/en_US.ts
  ${CMAKE_CURRENT_SOURCE_DIR}/lang/zh_CN.ts)
set_source_files_properties(${TS_FILES}
  PROPERTIES OUTPUT_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lang")
qt5_create_translation(QM_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TS_FILES})

add_executable(qt_translation
  qt-translation.cpp
  ${TS_FILES}
  ${QM_FILES})
target_link_libraries(qt_translation
  Qt5::Core
  Qt5::Widgets
  )

add_custom_command(
  TARGET qt_translation POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/lang ${CMAKE_CURRENT_BINARY_DIR}/lang
)