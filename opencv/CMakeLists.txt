cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

project(opencv_tutorial)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Qt5Widgets)


add_executable(cvCamera src/Camera.cpp)
target_link_libraries(cvCamera ${OpenCV_LIBRARIES})

add_executable(cvKernel src/Kernel.cpp)
target_link_libraries(cvKernel ${OpenCV_LIBRARIES})

add_executable(cvBlending src/Blending.cpp)
target_link_libraries(cvBlending ${OpenCV_LIBRARIES})

add_executable(cvContrast src/Contrast.cpp)
target_link_libraries(cvContrast ${OpenCV_LIBRARIES})

add_executable(cvDrawing src/Drawing.cpp)
target_link_libraries(cvDrawing ${OpenCV_LIBRARIES})

add_executable(cvDFT src/DFT.cpp)
target_link_libraries(cvDFT ${OpenCV_LIBRARIES})

add_executable(cvSmoothing src/Smoothing.cpp)
target_link_libraries(cvSmoothing ${OpenCV_LIBRARIES})

add_executable(cvEroding src/Eroding.cpp)
target_link_libraries(cvEroding ${OpenCV_LIBRARIES})

add_executable(cvImage src/Image.cpp)
target_link_libraries(cvImage ${OpenCV_LIBRARIES})

add_executable(cvRotate src/Rotate.cpp)
target_link_libraries(cvRotate ${OpenCV_LIBRARIES})

add_executable(cvDetectMotion src/DetectMotion.cpp)
target_link_libraries(cvDetectMotion ${OpenCV_LIBRARIES})

add_executable(cvInputOutputArray src/InputOutputArray.cpp)
target_link_libraries(cvInputOutputArray ${OpenCV_LIBRARIES})


# image processing
#add_executable(cvResize imgproc/Resize.cpp)
#target_link_libraries(cvResize ${OpenCV_LIBRARIES})

add_executable(cvMorphology imgproc/Morphology.cpp)
target_link_libraries(cvMorphology ${OpenCV_LIBRARIES})

add_executable(cvBoarder imgproc/Boarder.cpp)
target_link_libraries(cvBoarder ${OpenCV_LIBRARIES})

add_executable(cvPyramid imgproc/Pyramid.cpp)
target_link_libraries(cvPyramid ${OpenCV_LIBRARIES})

add_executable(cvSobel imgproc/Sobel.cpp)
target_link_libraries(cvSobel ${OpenCV_LIBRARIES})

add_executable(cvLaplace imgproc/Laplace.cpp)
target_link_libraries(cvLaplace ${OpenCV_LIBRARIES})

add_executable(cvCanny imgproc/CannyDetector.cpp)
target_link_libraries(cvCanny ${OpenCV_LIBRARIES})

add_executable(cvColorSpace imgproc/ColorSpace.cpp)
target_link_libraries(cvColorSpace ${OpenCV_LIBRARIES})

#add_executable(cvCCM imgproc/ColorCorrectionMatrix.cpp)
#target_link_libraries(cvCCM ${OpenCV_LIBRARIES})

add_executable(cvHoughLines imgproc/Houghlines.cpp)
target_link_libraries(cvHoughLines ${OpenCV_LIBRARIES})

add_executable(cvHoughCircles imgproc/Houghcircles.cpp)
target_link_libraries(cvHoughCircles ${OpenCV_LIBRARIES})

add_executable(cvEqualize imgproc/EqualizeHist.cpp)
target_link_libraries(cvEqualize ${OpenCV_LIBRARIES})

add_executable(cvBackProject imgproc/CalcBackProject.cpp)
target_link_libraries(cvBackProject ${OpenCV_LIBRARIES})

add_executable(cvCamshiftdemo imgproc/Camshiftdemo.cpp)
target_link_libraries(cvCamshiftdemo ${OpenCV_LIBRARIES})

add_executable(cvMatchTemplate imgproc/MatchTemplate.cpp)
target_link_libraries(cvMatchTemplate ${OpenCV_LIBRARIES})

add_executable(cvContours imgproc/Contours.cpp)
target_link_libraries(cvContours ${OpenCV_LIBRARIES})



# ===========================
# Highgui
# ===========================
add_executable(cvCallback gui/cvCallback.cpp)
target_link_libraries(cvCallback ${OpenCV_LIBRARIES})

add_executable(cvNamedWindow gui/cvNamedWindow.cpp)
target_link_libraries(cvNamedWindow ${OpenCV_LIBRARIES})


# ===========================
# Cuda
# ===========================
option(BUILD_CUDA_EXAMPLES "Turn ON to build CV CUDA examples" OFF)
if(BUILD_CUDA_EXAMPLES)
    add_subdirectory(cv_cuda)
endif(BUILD_CUDA_EXAMPLES)

file(COPY imgs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
