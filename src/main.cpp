#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include "MatrixPar.h"

#include <iostream>


int main(int argc, char **argv) {
    /*
    Matrix x_train;
    Matrix y_train;
    std::cout << "Preprocessing the data" << std::endl;
    readCSV("../data/mnist_train.csv", false, x_train, y_train);
    oneHotEncoding(y_train, 10);
    Matrix x_test;
    Matrix y_test;
    readCSV("../data/mnist_test.csv", false, x_test, y_test);
    oneHotEncoding(y_test, 10);
    //generateSinusData(x, y, 100);

    Optimizer* opti = new Adam(0.01, 0.9, 0.999, 1e-8);
    // Optimizer* opti = new SGD(1, 0.9);

    NeuralNetwork net(opti, "cross_entropy", "accuracy");

    net.addLayer(300, "relu", 784);
    net.addLayer(10, "softmax");
    // net.addLayer(5, "sigmoid", 1);
    // net.addLayer(5, "sigmoid");
    // net.addLayer(1, "relu");
    //
    // // net.load("../data/sinus_training.data");
    //
    std::cout << "Fitting the data" << std::endl;
    net.fit(x_train, y_train, 3, 128);

    std::cout << "Validating the model" << std::endl;
    net.validate(x_test, y_test);

    // //net.save("../data/sinus_training.data");
    // double input = -1.57/3; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    //std::cout << net << std::endl;

     */

    double start_time = omp_get_wtime();
    Matrix m1 = Matrix(500, 500);
    m1 = 1 - m1;
    Matrix m2 = Matrix(500, 500);
    m2 = 2 - m2;
    for (int i=0; i<10; i++) {
        m2 = m1 * m2;
    }
    double run_time = omp_get_wtime() - start_time;
    printf("\n Matrix seq multiplications in %lf seconds\n",run_time);
    std::cout << m2(0,0) << std::endl;

    start_time = omp_get_wtime();
    MatrixPar m3 = MatrixPar(500, 500);
    m3 = 1 - m3;
    MatrixPar m4 = MatrixPar(500, 500);
    m4 = 2 - m4;
    for (int i=0; i<10; i++) {
        m4 = m3 * m4;
    }
    run_time = omp_get_wtime() - start_time;
    printf("\n Matrix // multiplications in %lf seconds\n",run_time);
    std::cout << m4(0,0) << std::endl;

    return 0;
}
