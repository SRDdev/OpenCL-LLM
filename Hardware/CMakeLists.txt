cmake_minimum_required(VERSION 3.10)
project(GPGPU_Diagnostics)

set(CMAKE_CXX_STANDARD 17)

# Locate OpenCL on the system
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(cl_info src/cl_info.cpp)
target_link_libraries(cl_info OpenCL::OpenCL)

# Optional: Add compiler flags for optimization
if(MSVC)
    target_compile_options(cl_info PRIVATE /W4 /WX)
else()
    target_compile_options(cl_info PRIVATE -Wall -Wextra -O3)
endif()