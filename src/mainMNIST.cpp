#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>

#include <iostream>

int DEVICE = 0;

int main(int argc, char **argv) {

    Matrix x_train;
    Matrix y_train;
    std::cout << "Preprocessing the data" << std::endl;
    readCSV("../data/mnist_train.csv", false, x_train, y_train);
    oneHotEncoding(y_train, 10);
    Matrix x_test;
    Matrix y_test;
    readCSV("../data/mnist_test.csv", false, x_test, y_test);
    oneHotEncoding(y_test, 10);

    Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);

    NeuralNetwork net(opti, "cross_entropy", "accuracy");

    net.addLayer(300, "relu", 784);
    net.addLayer(150, "relu");
    net.addLayer(10, "softmax");

    std::cout << "Fitting the data" << std::endl;
    net.fit(x_train, y_train, 1, 128);

    std::cout << "Validating the model" << std::endl;
    net.validate(x_test, y_test);

    //std::cout << "Saving the model" << std::endl;
    //net.save("../data/mnist_model.data");

    return 0;
}
