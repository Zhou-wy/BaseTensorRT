cmake_minimum_required(VERSION 3.15)

project(TrtServer)

set(CMAKE_CXX_STANDARD 11)

#cmake在编译过程中打印编印信息
set(CMAKE_VERBOSE_MAKEFILEON ON) 

find_package(OpenCV)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_BUILD_TYPE Debug)
set(CUDA_HOME /usr/local/cuda)
set(TRT_HOME /home/zwy/TensorRT-7.2.3.4)

# 设置动态链接库路径
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)
set(TRT_LIB_DIR ${TRT_HOME}/lib)
set(LD_CUDA_LIBS cuda cudart)
set(LD_TRT_LIBS myelin nvcaffe_parser nvinfer nvinfer_plugin nvonnxparser nvparsers)

# 设置输出bin文件路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set()
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations")
# cuda 和 cudnn 头文件
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)

# TensorRT 头文件
include_directories(${TRT_HOME}/include)
include_directories(${TRT_HOME}/sample)

# OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# src 链接库
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/common)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/builder)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/TrtLib/infer)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/app_yolo)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/app_http)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/app_WebSocket)


set(EXTRA_LIBS ${EXTRA_LIBS}  WebSocket http yolo TrtInfer TrtBuilder common)

link_directories(${CUDA_LIB_DIR} ${TRT_LIB_DIR}) 
add_executable(test_yolo_http ${CMAKE_SOURCE_DIR}/test/test_yolo_http.cpp)
# 链接动态链接库
target_link_libraries(test_yolo_http ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS})

add_executable(test_websocket ${CMAKE_SOURCE_DIR}/test/test_websocket.cpp)
# 链接动态链接库
target_link_libraries(test_websocket ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${EXTRA_LIBS})
