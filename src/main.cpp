#include "NeuralNetwork.h"
#include "Matrix.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv) {
//    std::vector<std::vector<double> > x;
//    std::vector<double> x_test;
//    x_test.push_back(0);
//    std::vector<double> d;
//
//    NeuralNetwork *net = new NeuralNetwork(x, d, 100);
//
//    net->fit(x, d, 1000, 0.01);
//    double output = net->predict(x_test);
//    std::cout << "Output: " << output << std::endl;
//

    Matrix m(3,2);
    m(0,0) = 1; m(0,1) = 2;
    m(1,0) = 3; m(1,1) = 4;
    m(2,0) = 10; m(2,1) = 100;
    std::cout << m << std::endl;

    std::vector<double> x;
    x.push_back(1.); x.push_back(10.);
    std::vector<double> y = m * x;

    for (int i = 0; i < y.size(); i++) {
        std::cout << y[i] << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
