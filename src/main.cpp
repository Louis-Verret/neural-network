#include "NeuralNetwork.h"
#include "Utils.h"

#include <iostream>


int main(int argc, char **argv) {
    Matrix x;
    Matrix d;
    generateSinusData(x, d, 100);

    NeuralNetwork *net = new NeuralNetwork();

    std::string sigmoid ("sigmoid");
    net->addLayer(5, sigmoid, 1);
    net->addLayer(5, sigmoid);
    net->addLayer(1, sigmoid);

    //std::cout << *net;

    net->fit(x, d, 20000, 1, 32, 0.9);

    //net->save("../data/sinus_training.data");
    double input = 1.57; // pi/2
    Matrix x_test(1, 1);
    x_test(0, 0) = (input + 1.0) / 5.0;
    Matrix output = net->predict(x_test);
    std::cout << "Output: " << output(0,0)*2 -1 << std::endl;

    delete net;

    return 0;
}
