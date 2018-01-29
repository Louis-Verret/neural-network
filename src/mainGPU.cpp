#include "NeuralNetwork.h"
#include "Utils.h"
#include "Optimizer.h"
#include <omp.h>
#include <iostream>
#include "MatrixGPU.h"
#include "VectorGPU.h"
#include "../Common/err_code.h"

int DEVICE = 0;

int main(int argc, char **argv) {

    try {

        int n = 10;
        int k = 7;
        int m = 8;
        int iter = 1;

        // Vector v4(n);
        //Vector v5(n);
        Matrix m5 = Matrix(n, k);
        //Matrix m6 = Matrix(k, m);
        // // Matrix m7 = Matrix(m, n);
        //m5.fillWithZero();
        //std::cout << m5 << std::endl;
        m5.fillRandomly();
        // v4.fillRandomly();
        //v5.fillRandomly();

        double start_time;
        double time_mean = 0;
        for (int i = 0; i<iter; i++) {
            start_time = omp_get_wtime();
            Matrix m7 = m5.argmax();
            // Vector v6 = (3 * v4).sqrt();
            time_mean += omp_get_wtime() - start_time;
        }
        printf("\n Matri1000x CPU multiplications in %lf seconds\n", time_mean/iter);

        MatrixGPU m2(n, k);
        MatrixGPU m3(k, m);
        m2.fillRandomly();
        m3.fillRandomly();
        // MatrixGPU m3(m, n);
        // VectorGPU v1(n);
        // v1.fillRandomly();
        // VectorGPU v2(n);
        // VectorGPU v0(n);
        // std::cout << v1 << std::endl;
        //std::cout << v2 << std::endl;
        //std::cout << m3 << std::endl;
        // std::cout << (m2 * m3) << std::endl;
        // std::cout << ((m2 * m3) + m2) << std::endl;
        time_mean = 0;
        for (int i = 0; i<iter; i++) {
            start_time = omp_get_wtime();
            MatrixGPU m4 = m2 * m3;
            //VectorGPU v3 = (3 * v2).sqrt();
            time_mean += omp_get_wtime() - start_time;
            std::cout << m4 << std::endl;
        }
        // m4 = m4 * m3;
        //MatrixGPU m4 = m1.transpose();
        printf("\n Matrix GPU multiplications in %lf seconds\n", time_mean/iter);


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
