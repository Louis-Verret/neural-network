cmake_minimum_required (VERSION 2.8.11)

project(HPC_GPU)

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall -fopenmp -L/opt/amdgpu-pro/lib64")

set(SOURCES
        src/Matrix.cpp
        src/Vector.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/ActivationFunction.cpp
        src/Utils.cpp
        src/ProgressBar
        src/Optimizer.cpp
        src/CostFunction.cpp
        src/Metric.cpp

        src/GPU.cpp
        src/MatrixGPU.cpp
        src/VectorGPU.cpp
        )

link_directories(Common)

add_executable(gpu src/mainGPU.cpp ${SOURCES})
add_executable(sinus src/mainSinus.cpp ${SOURCES})
add_executable(mnist src/mainMNIST.cpp ${SOURCES})

target_link_libraries(gpu OpenCL rt)
target_link_libraries(sinus OpenCL rt)
target_link_libraries(mnist OpenCL rt)
