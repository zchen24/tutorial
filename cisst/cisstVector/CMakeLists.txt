# create a list of required cisst libraries
set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstNumerical
                              cisstOSAbstraction
                              )

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})

  add_executable (Vector mainVector.cpp)
  cisst_target_link_libraries (Vector ${REQUIRED_CISST_LIBRARIES})

endif (cisst_FOUND_AS_REQUIRED)
