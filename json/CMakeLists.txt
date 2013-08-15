cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(json_tutorial)

# pkg-config
find_package(PkgConfig REQUIRED)
if(PKG_CONFIG_FOUND)
  message(STATUS "PkgConfig found: " ${PKG_CONFIG_EXECUTABLE} )

  pkg_check_modules(JSON REQUIRED jsoncpp)
  if(JSON_FOUND)
    
    # include header dir
    include_directories(${JSON_INCLUDE_DIRS})

    # now add example
    add_executable(example src/example.cpp)
    target_link_libraries(example ${JSON_LIBRARIES})
    
  endif(JSON_FOUND)
  
endif(PKG_CONFIG_FOUND)

