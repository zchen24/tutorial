cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cmake-examples)

# Enable C++11
add_compile_options(-std=c++11)

# Search FindXXX.cmake in cmake folder
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules")

# Find package
# find_package()

# Add include directory 
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add link directory
link_directories(${CMAKE_SOURCE_DIR}/libs)

# set(HEADERS )
file(GLOB HEADERS include/*.h)
message(STATUS "headers: " ${HEADERS})

# executable
add_executable(cmake-main
    main.cpp
    ${HEADERS}
    MyClass.cpp)
# target_link_libraries(cmake-main OTHERLIB)

# add sub-directory
add_subdirectory(subdir)

# --------------------------------
# Install
# --------------------------------
install(TARGETS cmake-main
    RUNTIME
    DESTINATION bin
    COMPONENT applications
    )


# === uncomment to include system library 
# set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
# include(InstallRequiredSystemLibraries)
# install(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
#     DESTINATION bin
#     COMPONENT applications)

# --------------------------------
# Windows NSIS Installer
# --------------------------------
set(CPACK_PACKAGE_NAME "CMakeExamples")
set(CPACK_PACKAGE_VENDOR "Zihan Chen")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A package shows how to use cmake")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMakeExamples")

# tell CPack about the components
set(CPACK_COMPONENTS_ALL applications)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")


if(WIN32)
    message(STATUS "Windows specific CPACK")
    # Icons
    # set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}\\\\hw.bmp")
    # set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/hw.ico")
    # set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/hw.ico")
    set(CPACK_NSIS_CONTACT "zihan.chen.jhu@gmail.com")
    set(CPACK_NSIS_MENU_LINKS "bin\\cmake-main" "CMakeExamples")
else(WIN32)
    message(STATUS "Non Windows")
    # TODO create a debian package
endif(WIN32)

include(CPack)
