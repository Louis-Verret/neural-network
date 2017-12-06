cmake_minimum_required (VERSION 2.8.11)

project(HPC_GPU)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")

set(SOURCES
        src/Matrix.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        )

add_executable(neuralnet src/main.cpp ${SOURCES})
