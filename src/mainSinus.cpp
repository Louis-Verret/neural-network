#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>

#include <iostream>


int main(int argc, char **argv) {

    /* Preprocessing (generating data)*/
    Matrix x_train;
    Matrix y_train;
    generateSinusData(x_train, y_train, 1000);

    /* Constructing the neural network (layers, optimizer ..)*/
    Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    //Optimizer* opti = new SGD(0.1, 0.9);

    NeuralNetwork net(opti, "mean_squared_error");

    net.addLayer(5, "sigmoid", 1);
    net.addLayer(5, "sigmoid");
    net.addLayer(1, "relu");

    /** Loading the model **/
    // NeuralNetwork net(opti, "mean_squared_error");
    // net.load("../model/sinus.model");

    /* Training the model */
    std::cout << "Building the model" << std::endl;
    net.fit(x_train, y_train, 10000, 128);

    /* Validating the model */
    double input = -1.57/3; // pi/6
    Matrix x_test(1, 1);
    x_test(0, 0) = (input + 4)/8;
    Matrix output = net.predict(x_test);
    std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    /* Saving the model */
    std::cout << "Saving the model" << std::endl;
    net.save("../model/sinus.model");

    return 0;
}
