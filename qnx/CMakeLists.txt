cmake_minimum_required(VERSION 3.0)
project("qnx_example")

add_executable(ex_fork ex_fork.cpp)

add_executable(ex_thread_create ex_thread_create.cpp)
add_executable(ex_thread_join ex_thread_join.cpp)
add_executable(ex_thread_barrier ex_thread_barrier.cpp)


add_executable(ex_msg_client ex_msg_client.cpp)
add_executable(ex_msg_server ex_msg_server.cpp)

add_executable(ex_sched ex_sched.cpp)
add_executable(ex_fcntl ex_fcntl.cpp)
add_executable(ex_timer ex_timer.cpp)
add_executable(ex_clock ex_clock.cpp)
add_executable(ex_kernel_timeout ex_kernel_timeout.cpp)
