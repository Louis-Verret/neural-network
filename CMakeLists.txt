cmake_minimum_required (VERSION 2.8.11)

project(HPC_GPU)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")

add_subdirectory(src)

add_executable(philips src/main.cpp)
