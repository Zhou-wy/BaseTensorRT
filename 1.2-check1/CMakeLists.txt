cmake_minimum_required(VERSION 3.20)
set(PROJECTNAME check)
project(${PROJECTNAME})

set(CUDA_HOME /usr/local/cuda)
set(CUDA_LIBS cuda)

include_directories(${CUDA_HOME}/include)

link_directories(${CUDA_HOME}/lib)

add_executable(${PROJECTNAME}  main.cpp)
target_link_libraries(${PROJECTNAME} ${CUDA_LIBS})
