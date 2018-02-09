#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include "../Common/err_code.h"

#include <iostream>

int DEVICE = 0;

int main(int argc, char **argv) {

    try {
        MatrixCPU x_train;
        MatrixCPU y_train;
        std::cout << "Preprocessing the data" << std::endl;
        readCSV("../data/mnist_train.csv", false, x_train, y_train);
        centralizeData(x_train, 0, 255);
        oneHotEncoding(y_train, 10);
        MatrixCPU x_test;
        MatrixCPU y_test;
        readCSV("../data/mnist_test.csv", false, x_test, y_test);
        centralizeData(x_test, 0, 255);
        oneHotEncoding(y_test, 10);

        // Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
        Optimizer* opti = new SGD(0.001, 0.9);

        NeuralNetwork net(opti, "cross_entropy", "accuracy");

        net.addLayer(10, "softmax", 784);

        std::cout << "Fitting the data" << std::endl;
        net.fit(x_train, y_train, 1, 128);

    } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return 0;
}
