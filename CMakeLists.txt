cmake_minimum_required (VERSION 2.8.11)

project(HPC_GPU)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -fopenmp")

set(SOURCES
        src/Matrix.cpp
        src/Vector.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/ActivationFunction.cpp
        src/Utils.cpp
        src/Optimizer.cpp
        src/CostFunction.cpp
        src/Metric.cpp
        src/MatrixPar.cpp)

add_executable(neuralnet src/main.cpp ${SOURCES})
