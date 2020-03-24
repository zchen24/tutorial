# create a list of required cisst libraries
set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstNumerical
                              cisstOSAbstraction
                              cisstRobot
                              )

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})
  cisst_find_saw_component (sawControllers REQUIRED)

  add_executable (LinearRn mainLinearRn.cpp)
  cisst_target_link_libraries (LinearRn ${REQUIRED_CISST_LIBRARIES})

  add_executable (QuinticRn mainQuinticRn.cpp)
  cisst_target_link_libraries (QuinticRn ${REQUIRED_CISST_LIBRARIES})

  add_subdirectory(kdl)

endif (cisst_FOUND_AS_REQUIRED)
