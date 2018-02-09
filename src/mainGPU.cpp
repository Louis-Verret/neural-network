#include <omp.h>
#include <iostream>
#include "VectorGPU.h"
#include "../Common/err_code.h"
#include "Matrix.h"
#include "MatrixCPU.h"

int DEVICE = 0;

int main(int argc, char **argv) {

    try {

        /** INIT */

        int n = 3000;
        int k = 3000; //576
        int m = 3000;
        int iter = 1;

        srand(time(NULL));

        std::vector<double> random_vec_1 (n * k);
        std::vector<double> random_vec_2 (k * m);
        for (int i = 0; i < n * k; i++)
            random_vec_1[i] = ((double) rand() / (double) RAND_MAX);
        for (int i = 0; i < k * m; i++)
            random_vec_2[i] = ((double) rand() / (double) RAND_MAX);


        /** CPU **/

        MatrixCPU m0(n, k);
        MatrixCPU m1(k, m);
        MatrixCPU m2(n, m);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                m0(i, j) = random_vec_1[i * k + j];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < m; j++)
                m1(i, j) = random_vec_1[i * m + j];

        double start_time;
        double time_mean = 0;
        for (int i = 0; i<iter; i++) {
            start_time = omp_get_wtime();
            m0 = m0 + m1;
            time_mean += omp_get_wtime() - start_time;
        }

        printf("\n Matrix CPU multiplications in %lf seconds\n", time_mean/iter);


        /** GPU **/

        Matrix m3(n, k);
        Matrix m4(k, m);
        Matrix m5(n, m);

        std::vector<double> m3_vec(m3.getPaddingM() * m3.getPaddingN());
        std::vector<double> m4_vec(m4.getPaddingM() * m4.getPaddingN());

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                m3_vec[i * m3.getPaddingM() + j] = random_vec_1[i * k + j];


        for (int i = 0; i < k; i++)
            for (int j = 0; j < m; j++)
                m4_vec[i * m4.getPaddingM() + j] = random_vec_1[i * m + j];

        cl::Buffer m3_buffer = cl::Buffer(GPU::context, m3_vec.begin(), m3_vec.end(), true);
        cl::Buffer m4_buffer = cl::Buffer(GPU::context, m4_vec.begin(), m4_vec.end(), true);
        m3.setBuffer(m3_buffer);
        m4.setBuffer(m4_buffer);

        time_mean = 0;
        for (int i = 0; i<iter; i++) {
            start_time = omp_get_wtime();
            m3 = m3 + m4;
            time_mean += omp_get_wtime() - start_time;
        }
        printf("\n Matrix GPU multiplications in %lf seconds\n", time_mean/iter);

        std::vector<double> mat_copy(m3.getPaddingN()*m3.getPaddingM());
        cl::copy(GPU::queue, m3.getBuffer(), mat_copy.begin(), mat_copy.end());


        /** Checking the results */

        bool all_same = true;
        double eps = 1e-7;
        for (int i = 0; i < m3.getN(); i++) {
            for (int j = 0; j < m3.getM(); j++) {
                if (abs(mat_copy[i * m3.getPaddingM() + j] - m0(i, j)) > eps) {
                    all_same = false;
                    break;
                }
            }
        }
        if (all_same) {
            std::cout << "\n CORRECT results" << std::endl;
        } else {
            std::cout << "\n INCORRECT results" << std::endl;
        }



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
