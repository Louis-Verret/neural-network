#include "NeuralNetwork.h"
#include "Utils.h"

#include <iostream>


int main(int argc, char **argv) {
    std::vector<std::vector<double> > x;
    std::vector<std::vector<double> > d;
    //std::vector<double> x_test;
    //x_test.push_back(0);

    NeuralNetwork *net = new NeuralNetwork();
    generateSinusData(x, d, 100);

    std::string sigmoid ("sigmoid");
    net->addLayer(5, sigmoid, 1);
    net->addLayer(5, sigmoid);
    net->addLayer(1, sigmoid);

    //std::cout << *net;

    //net->fit(x, d, 10000, 0.1);

    net->save("../data/sinus_training.data");

    //double output = net->predict(x_test)[0];
    //std::cout << "Output: " << output << std::endl;


    return 0;
}
