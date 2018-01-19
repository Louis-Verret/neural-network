#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include <iostream>
// #include "MatrixGPU.h"


int main(int argc, char **argv) {

    Matrix x_train;
    Matrix y_train;
    std::cout << "Preprocessing the data" << std::endl;
    readCSV("../data/mnist_train.csv", false, x_train, y_train);
    centralizeData(x_train, 0, 255);
    oneHotEncoding(y_train, 10);
    Matrix x_test;
    Matrix y_test;
    readCSV("../data/mnist_test.csv", false, x_test, y_test);
    centralizeData(x_test, 0, 255);
    oneHotEncoding(y_test, 10);
    //generateSinusData(x_train, y_train, 100);

    Optimizer* opti = new Adam(0.001, 0.9, 0.999, 1e-8);
    //Optimizer* opti = new SGD(0.1, 0.9);

    NeuralNetwork net(opti, "cross_entropy", "accuracy");
    //
    net.addLayer(300, "relu", 784);
    net.addLayer(150, "relu");
    net.addLayer(10, "softmax");
    // // net.addLayer(5, "sigmoid", 1);
    // // net.addLayer(5, "sigmoid");
    // // net.addLayer(1, "relu");
    // //
    // //net.load("../data/mnist_model.data");
    // //
    std::cout << "Building the model" << std::endl;
    net.fit(x_train, y_train, 1, 128);

    std::cout << "Validating the model" << std::endl;
    net.validate(x_test, y_test);

    //std::cout << "Saving the model" << std::endl;
    //net.save("../data/mnist_model.data");
    // double input = -1.57/3; // pi/2
    // Matrix x_test(1, 1);
    // x_test(0, 0) = (input + 4)/8;
    // Matrix output = net.predict(x_test);
    // std::cout << "sin(-pi/6): " << 2*output(0,0) -1 << std::endl;

    //std::cout << net << std::endl;



    // int n = 1000;
    // int iter = 200;
    //
    // Matrix m1 = Matrix(n, n);
    // Matrix m2 = Matrix(n, n);
    // m1.fillWithZero();
    // m2.fillWithZero();
    // m2 = m2 + 1;
    // double mean = 0;
    // for (int i = 0; i<iter; i++) {
    //     double start_time = omp_get_wtime();
    //     Matrix m9 = m1 + m2;
    //     double run_time = omp_get_wtime() - start_time;
    //     mean += run_time;
    // }
    // printf("\n Matrix seq multiplications in %lf seconds\n",mean/iter);
    // //
    // //
    // // MatrixPar m3 = MatrixPar(n, n);
    // // MatrixPar m4 = MatrixPar(n, n);
    // // m3.fillWithZero();
    // // m4.fillWithZero();
    // // m4 = m4 + 1;
    // // start_time = omp_get_wtime();
    // // MatrixPar m8 = m4.argmax();
    // // run_time = omp_get_wtime() - start_time;
    // // printf("\n Matrix // multiplications in %lf seconds\n",run_time);


 /*   // DEVICE is defined in matrixGPU.h as a global var
    cl::Context context(DEVICE);
    cl::CommandQueue queue(context, DEVICE);

    // Kernels compil
    cl::Program program_mul(context, util::loadProgram("matmul.cl"), true);
    cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program_mul, "mmul");

    MatrixGPU m1(context, 1024, 1024);
    MatrixGPU m2(context, 1024, 1024);
    MatrixGPU m4(context, 1024, 1024);
    m4 = m1.matmult(queue, naive_mmul, m2);*/


    return 0;
}
