cmake_minimum_required (VERSION 2.6.2)

project(cisst_example)

# set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

# cisstVector
add_subdirectory(cisstVector)

# cisstRobot
add_subdirectory(cisstRobot)

# mtsTutorial
add_subdirectory(mtsTutorial)

# mtsPreemprtRT
add_subdirectory(mtsPreemptRT)

# svlTutorial
#add_subdirectory(svlTutorial)


