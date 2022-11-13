cmake_minimum_required(VERSION 3.17)
set(PROJECT_NAME cumemory)
project(${PROJECT_NAME})

set(CUDA_HOME /usr/local/cuda)
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)

set(LD_LIBS cuda cudart)
include_directories(${CUDA_HOME}/include)
link_directories(${CUDA_LIB_DIR}) 

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${LD_LIBS})




