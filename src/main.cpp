#include "NeuralNetwork.h"
#include "Matrix.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv) {
   std::vector<std::vector<double> > x;
   std::vector<double> d;
   //std::vector<double> x_test;
   //x_test.push_back(0);

   NeuralNetwork *net = new NeuralNetwork(x, d, 100);

   std::string sigmoid ("sigmoid");
   net->addLayer(2, sigmoid, 1);
   net->addLayer(1, sigmoid);
   net->fit(x, d, 1000, 0.01);
   //double output = net->predict(x_test);
   //std::cout << "Output: " << output << std::endl;


    return 0;
}
