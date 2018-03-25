cmake_minimum_required(VERSION 2.8)

# executable
add_executable(cmake-main-subdir
    main-subdir.cpp)

# --------------------------------
# Install
# --------------------------------
install(TARGETS cmake-main-subdir
    RUNTIME
    DESTINATION bin
    COMPONENT applications
    )
