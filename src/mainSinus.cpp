#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>

#include <iostream>

int DEVICE = 0;

int main(int argc, char **argv) {

    MatrixCPU x_train;
    MatrixCPU y_train;
    generateSinusData(x_train, y_train, 10000);

    Optimizer* opti = new SGD(0.001, 0.9);

    NeuralNetwork net(opti, "mean_squared_error");

    net.addLayer(5, "sigmoid", 1);
    net.addLayer(5, "sigmoid");
    net.addLayer(1, "sigmoid");

    std::cout << "Fitting the data" << std::endl;
    net.fit(x_train, y_train, 10, 100);

    return 0;
}
