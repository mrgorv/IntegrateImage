cmake_minimum_required(VERSION 3.1.0)
project(IntegralImage)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif ()

set(CMAKE_CXX_STANDARD 14)
add_executable(integral_image
    main.cpp
    integral_image.h
    integral_image.cpp
    )
    
ADD_SUBDIRECTORY(tests)
target_link_libraries(integral_image ${OpenCV_LIBS})
