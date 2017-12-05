#include "NeuralNetwork.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {


    std::vector<std::vector<double> > x;
    std::vector<double> x_test;
    x_test.push_back(0);
    std::vector<double> d;

    NeuralNetwork* net = new NeuralNetwork(x, d, 100);

    net->fit(x, d, 1000, 0.01);
    double output = net->predict(x_test);
    std::cout << "Output: " << output << std::endl;

    return 0;

}
