project(HPC_GPU)

set(SOURCES
        Matrix.cpp
        NeuralNetwork.cpp
        )

add_library(${PROJECT_NAME} ${SOURCES})
