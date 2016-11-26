cmake_minimum_required(VERSION 2.8.8)

# Project name
project(disparity)

# Include our cmake files
# SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Include libraries
#find_package(Eigen3 REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system serialization system filesystem thread program_options date_time regex timer chrono)
# find_package(OpenCV 3 REQUIRED core imgcodecs videoio ximgproc)


# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Enable debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3  -Wall")


# Include our header files
include_directories(src)

# Include all thirdparty source files in build
# file(GLOB_RECURSE sources_thirdparty thirdparty/*.cpp)

#####################################
# Main class files for TOTO
#####################################
set(sources_cpu
    src/main.cpp
    src/lodepng.cpp
)



# Create the main executable, and link libraries
add_executable(stereo_cpu ${sources_cpu})

# Link main libraries
#target_link_libraries(stereo_cpu ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

