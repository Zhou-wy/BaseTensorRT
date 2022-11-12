cmake_minimum_required(VERSION 3.15)
set(ProjectName ffhdd)
project(${ProjectName})

# # 第三方库
set(CUDA_HOME /usr/local/cuda)
set(TRT_HOME /home/zwy/TensorRT-7.2.3.4)
enable_language(CUDA)
find_package(OpenCV)
# 设置ffmpeg依赖库及头文件所在目录，并存进指定变量
set(ffmpeg_libs_DIR /usr/lib/x86_64-linux-gnu)
set(ffmpeg_headers_DIR /usr/include/x86_64-linux-gnu)


add_library( avcodec SHARED IMPORTED )
 
#指定所添加依赖库的导入路径
set_target_properties( avcodec PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libavcodec.so )
 
add_library( avfilter SHARED IMPORTED )
set_target_properties( avfilter PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libavfilter.so )
 
add_library( swresample SHARED IMPORTED )
set_target_properties( swresample PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libswresample.so )
 
add_library( swscale SHARED IMPORTED )
set_target_properties( swscale PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libswscale.so )
 
add_library( avformat SHARED IMPORTED )
set_target_properties( avformat PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libavformat.so )
 
add_library( avutil SHARED IMPORTED )
set_target_properties( avutil PROPERTIES IMPORTED_LOCATION ${ffmpeg_libs_DIR}/libavutil.so )


# CUDA and cudnn include dir 
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)

# TensorRT include dir
include_directories(${TRT_HOME}/include)
include_directories(${TRT_HOME}/samples)

# OpenCV include dir
include_directories(${OpenCV_INCLUDE_DIRS})

# ffmpeg include dir 
include_directories(${ffmpeg_headers_DIR})

# Personal Src include 
include_directories(${CMAKE_SOURCE_DIR}/src/TrtLib)

# CUDA library dir
set(CUDA_LIB_DIR ${CUDA_HOME}/lib64)

# TensorRT library dir
set(TRT_LIB_DIR ${TRT_HOME}/lib)

# CUDA libs
set(LD_CUDA_LIBS cuda cudart)

# TensorRT libs
set(LD_TRT_LIBS myelin nvinfer nvinfer_plugin nvonnxparser nvparsers)
# ffmpeg libs
set(LD_FFMPEG_LIBS  avcodec avformat avutil swresample swscale swscale avfilter)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.c *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

link_directories(${CUDA_LIB_DIR}) 
link_directories(${TRT_LIB_DIR}) 
link_directories(${ffmpeg_libs_DIR})

# create static library
add_library(${ProjectName} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${LD_TRT_LIBS} ${LD_CUDA_LIBS} ${OpenCV_LIBS} ${LD_FFMPEG_LIBS})
