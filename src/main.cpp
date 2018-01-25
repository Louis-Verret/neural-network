#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include <iostream>
#include "MatrixGPU.h"
#include "../Common/err_code.h"

int DEVICE = 0;

int main(int argc, char **argv) {

    // Matrix x_train;
    // Matrix y_train;
    // std::cout << "Preprocessing the data" << std::endl;
    // readCSV("../data/mnist_train.csv", false, x_train, y_train);
    // centralizeData(x_train, 0, 255);
    // oneHotEncoding(y_train, 10);
    // Matrix x_test;
    // Matrix y_test;
    // readCSV("../data/mnist_test.csv", false, x_test, y_test);
    // centralizeData(x_test, 0, 255);
    // oneHotEncoding(y_test, 10);
    // //generateSinusData(x_train, y_train, 100);
    //
    // Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    // //Optimizer* opti = new SGD(0.1, 0.9);
    //
    // NeuralNetwork net(opti, "cross_entropy", "accuracy");
    // //
    // net.addLayer(300, "relu", 784);
    // net.addLayer(150, "relu");
    // net.addLayer(10, "softmax");
    // // // net.addLayer(5, "sigmoid", 1);
    // // // net.addLayer(5, "sigmoid");
    // // // net.addLayer(1, "relu");
    // // //
    // // //net.load("../data/mnist_model.data");
    // // //
    // std::cout << "Building the model" << std::endl;
    // net.fit(x_train, y_train, 1, 128);
    //
    // std::cout << "Validating the model" << std::endl;
    // net.validate(x_test, y_test);

    //std::cout << "Saving the model" << std::endl;
    //net.save("../data/mnist_model.data");
    // double input = -1.57/3; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    //std::cout << net << std::endl;


    try {

        int n = 10000;
        int k = 20000;
        int m = 16;

        Matrix m5 = Matrix(n, k);
        // Matrix m6 = Matrix(k, m);
        // Matrix m7 = Matrix(m, n);
        m5.fillRandomly();
        // m6.fillRandomly();
        double start_time = omp_get_wtime();
        Matrix m9 = m5.transpose();
        double run_time = omp_get_wtime() - start_time;
        printf("\n Matrix CPU multiplications in %lf seconds\n", run_time);

        GPU::init();
        MatrixGPU m1(n, k);
        // MatrixGPU m2(k, m);
        // MatrixGPU m3(m, n);
        //std::cout << m1 << std::endl;
        start_time = omp_get_wtime();
        //MatrixGPU m4 = m1 * m2;
        // m4 = m4 * m3;
        MatrixGPU m4 = m1.transpose();
        run_time = omp_get_wtime() - start_time;
        //std::cout << m4 << std::endl;
        printf("\n Matrix GPU multiplications in %lf seconds\n", run_time);


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
