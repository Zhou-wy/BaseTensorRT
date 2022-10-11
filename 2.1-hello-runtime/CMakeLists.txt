cmake_minimum_required(VERSION 3.10)
set(PROJECTNAME hello_runtime)
project(${PROJECTNAME})

set(CUDA_HOME /usr/local/cuda)
set(CUDA_LIBS cuda cudart)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)

include_directories(${CUDA_HOME}/include)

link_directories(${CUDA_HOME}/lib64)

add_executable(${PROJECTNAME}  main.cpp)
target_link_libraries(${PROJECTNAME} ${CUDA_LIBS})
