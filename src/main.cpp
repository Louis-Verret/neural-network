#include "NeuralNetwork.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {


    std::vector<std::vector<double> > x;
    std::vector<double> d;

    double total_d = 1400.0 + 1600.0 + 1700.0 + 1875.0 + 1110.0 + 1550.0 + 2350.0 + 2450.0 + 1425.0;
    double total_x = 245.0 + 312.0 + 279.0 + 308.0 + 199.0 + 219.0 + 405.0 + 324.0 + 319.0;

    std::vector<double> x0;
    x0.push_back(245.0 / total_x);
    d.push_back(1400.0 / total_d);
    std::vector<double> x1;
    x1.push_back(312.0 / total_x);
    d.push_back(1600.0 / total_d);
    std::vector<double> x2;
    x2.push_back(279.0 / total_x);
    d.push_back(1700.0 / total_d);
    std::vector<double> x3;
    x3.push_back(308.0 / total_x);
    d.push_back(1875.0 / total_d);
    std::vector<double> x4;
    x4.push_back(199.0 / total_x);
    d.push_back(1110.0 / total_d);
    std::vector<double> x5;
    x5.push_back(219.0 / total_x);
    d.push_back(1550.0 / total_d);
    std::vector<double> x6;
    x6.push_back(405.0 / total_x);
    d.push_back(2350.0 / total_d);
    std::vector<double> x7;
    x7.push_back(324.0 / total_x);
    d.push_back(2450.0 / total_d);
    std::vector<double> x8;
    x8.push_back(319.0 / total_x);
    d.push_back(1425.0 / total_d);
    std::vector<double> x9;
    x9.push_back(255.0 / total_x);
    d.push_back(1700.0 / total_d);

    x.push_back(x0);
    x.push_back(x1);
    x.push_back(x2);
    x.push_back(x3);
    x.push_back(x4);
    x.push_back(x5);
    x.push_back(x6);
    x.push_back(x7);
    x.push_back(x8);


    NeuralNetwork* net = new NeuralNetwork(10);
    net->fit(x, d, 10000, 0.1);
    double output = net->predict(x9) * total_d;
    std::cout << "Output: " << output << std::endl;

    return 0;

}
