#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include <iostream>
#include "MatrixGPU.h"
#include "../Common/err_code.h"

int DEVICE = 0;

int main(int argc, char **argv) {

    try {

        int n = 2;
        int k = 5;
        int m = 3;

        Matrix m5 = Matrix(n, k);
        Matrix m6 = Matrix(k, m);
        Matrix m7 = Matrix(m, n);
        m5.fillRandomly();
        m6.fillRandomly();
        double start_time = omp_get_wtime();
        Matrix m9 = m5 * m6 * m7;
        double run_time = omp_get_wtime() - start_time;
        printf("\n Matrix CPU multiplications in %lf seconds\n", run_time);

        GPU::init();
        MatrixGPU m1(n, k);
        MatrixGPU m2(k, m);
        MatrixGPU m3(m, n);
        start_time = omp_get_wtime();
        MatrixGPU m4 = m1 * m2;
        m4 = m4 * m3;
        run_time = omp_get_wtime() - start_time;
        std::cout << m4 << std::endl;
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
