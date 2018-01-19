#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>

#include <iostream>


int main(int argc, char **argv) {

    Matrix x_train;
    Matrix y_train;
    generateSinusData(x_train, y_train, 100);

    Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    //Optimizer* opti = new SGD(0.1, 0.9);

    NeuralNetwork net(opti, "mean_squared_error");

    net.addLayer(5, "sigmoid", 1);
    net.addLayer(5, "sigmoid");
    net.addLayer(1, "relu");

    std::cout << "Fitting the data" << std::endl;
    net.fit(x_train, y_train, 10000, 10);


    double input = -1.57/3; // pi/2
    Matrix x_test(1, 1);
    x_test(0, 0) = (input + 4)/8;
    Matrix output = net.predict(x_test);
    std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    return 0;
}
