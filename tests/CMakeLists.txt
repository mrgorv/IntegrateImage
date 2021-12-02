cmake_minimum_required(VERSION 3.1)
 
# Locate GTest
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED PATHS "/home/egor/Sources/opencv/build/")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(runTests
    tests.cpp
    ../integral_image.h
    ../integral_image.cpp)
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)
target_link_libraries(runTests ${OPENCV_LIBS}
    "/home/egor/Sources/opencv/build/lib/libopencv_core.so.4.5.4"
    "/home/egor/Sources/opencv/build/lib/libopencv_imgcodecs.so"
    "/home/egor/Sources/opencv/build/lib/libopencv_highgui.so"
)
