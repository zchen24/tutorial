cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(CUDAFE LANGUAGES CXX CUDA)   # CUDA for Engineers
set (CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# chapter 0
add_subdirectory(printf)        # device printf
add_subdirectory(device_property)
add_subdirectory(per_stream)

add_subdirectory(dist_v1)
add_subdirectory(dist_v1_cuda)
add_subdirectory(dist_v2)
add_subdirectory(dist_v2_cuda)
add_subdirectory(dist_v2_cuda_unified)

# chapter 4
add_subdirectory(dist_2d)       # 2d

# chapter 5
add_subdirectory(dd_1d_global)  # global 
add_subdirectory(dd_1d_shared)  # shared_mem
add_subdirectory(sharpen)       # stencil

# chapter 6
add_subdirectory(parallel_dot)
add_subdirectory(centroid_2d)



# chapter 8: cuda lib
add_subdirectory(norm)          # thrust: inner_product
add_subdirectory(dist_1d_thrust)
add_subdirectory(curand_pi)     # curand
add_subdirectory(sharpen_npp)   # npp: sharpen
add_subdirectory(tricoloring)   # npp: swap, element add
add_subdirectory(linreg)        # cublas, cusolver



#------------------------
# CUDA by Example 
#------------------------

find_package(OpenCV)
if (OpenCV_FOUND)

# chapter 4
#add_subdirectory(04_julia)

# chapter 5
# add_subdirectory(04_julia)

endif (OpenCV_FOUND)

# chapter 10
add_subdirectory(10_stream)
