cmake_minimum_required(VERSION 3.10)
set(PROJECTNAME cumalloc)
project(${PROJECTNAME})

set(CUDA_HOME /usr/local/cuda)
set(CUDA_LIBS cuda)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)

include_directories(${CUDA_HOME}/include)
link_directories(${CUDA_HOME}/lib)

add_executable(${PROJECTNAME}  main.cpp)
target_link_libraries(${PROJECTNAME} ${CUDA_LIBS})
