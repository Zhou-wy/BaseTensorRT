# CMakeLists.txt for G4CU project
set(ProjectName TrtFunc)
project(${ProjectName})

# required cmake version
cmake_minimum_required(VERSION 3.15)

set(CUDA_HOME /usr/local/cuda)
set(TRT_HOME /home/zwy/TensorRT-7.2.3.4)
enable_language(CUDA)
find_package(OpenCV)

#include_directories ("${PROJECT_SOURCE_DIR}")
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)
include_directories(${TRT_HOME}/include)
include_directories(${TRT_HOME}/samples)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)
set(TRT_LIB_DIR ${TRT_HOME}/lib)
set(LD_CUDA_LIBS cuda cudart)
set(LD_TRT_LIBS myelin nvinfer nvinfer_plugin nvonnxparser nvparsers)
# nvcc flags -g for debug
# set(CUDA_NVCC_FLAGS -O3;-G;-g)

# set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

link_directories(${CUDA_LIB_DIR}) 
link_directories(${TRT_LIB_DIR}) 

#cuda_add_library(gpu SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# cuda_add_library(cudaFunc STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(${ProjectName} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS})