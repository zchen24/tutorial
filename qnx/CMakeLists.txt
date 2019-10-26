cmake_minimum_required(VERSION 3.0)
project("qnx_example")

set(EXAMPLES
  ex_fork

  ex_thread_create
  ex_thread_join
  ex_thread_barrier

  ex_msg_client
  ex_msg_server

  ex_sched
  ex_fcntl
  ex_timer
  ex_clock
  ex_kernel_timeout
  )

message("CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    target_link_libraries(${example} pthread)
  endif()
endforeach()
