#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"

#include <iostream>


int main(int argc, char **argv) {
    Matrix x;
    Matrix y;
    std::cout << "Preprocessing the data" << std::endl;
    readCSV("../data/mnist_train.csv", false, x, y);
    oneHotEncoding(y, 10);

    //generateSinusData(x, y, 100);

    Optimizer* opti = new Adam(0.01, 0.9, 0.999, 1e-8);
    // Optimizer* opti = new SGD(1, 0.9);

    NeuralNetwork net(opti, "cross_entropy", "accuracy");

    net.addLayer(10, "softmax", 784);
    // net.addLayer(5, "sigmoid", 1);
    // net.addLayer(5, "sigmoid");
    // net.addLayer(1, "relu");
    //
    // // net.load("../data/sinus_training.data");
    //
    std::cout << "Fitting the data" << std::endl;
    net.fit(x, y, 3, 128);

    // //net.save("../data/sinus_training.data");
    // double input = -1.57/3; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    //std::cout << net << std::endl;

    return 0;
}
