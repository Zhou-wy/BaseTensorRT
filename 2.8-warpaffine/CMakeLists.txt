cmake_minimum_required(VERSION 3.15)
set(PROJECT_NAME warpaffine)
project(${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

set(CMAKE_BUILD_TYPE Debug)
set(CUDA_HOME /usr/local/cuda)
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)
set(LD_LIBS cuda cudart)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(cudaFunc)

link_directories(${CUDA_LIB_DIR}) 
add_subdirectory(cudaFunc)
set(EXTRA_LIBS ${EXTRA_LIBS} cudaFunc)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${LD_LIBS} ${EXTRA_LIBS} ${OpenCV_LIBS})




