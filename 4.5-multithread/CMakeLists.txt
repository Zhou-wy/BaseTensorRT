cmake_minimum_required(VERSION 3.15.0)
set(ProJectName multiThread)
project(${ProJectName})


find_package(Threads)
add_executable(${ProJectName} main.cpp infer.cpp)
target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})