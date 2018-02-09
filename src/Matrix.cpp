#include <stdexcept>
#include "Matrix.h"
#include <cmath>

Matrix::Matrix() : m_padding_n(0), m_padding_m(0), m_n(0), m_m(0) {
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 0 * 0);
}

Matrix::~Matrix() {
}

Matrix::Matrix(int n, int m) : m_n(n), m_m(m) {
    int block_size = 32;
    int padding_n = (m_n%block_size != 0)? m_n + block_size - m_n%block_size : m_n;
    int padding_m = (m_m%block_size != 0)? m_m + block_size - m_m%block_size : m_m;
    int max_padding = std::max(padding_n, padding_m);
    m_padding_n = max_padding;
    m_padding_m = max_padding;
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * m_padding_n * m_padding_m);
}

Matrix Matrix::operator*(const Matrix &mat) const {

    if (mat.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    Matrix res(m_n, mat.getM());

    cl::NDRange global(m_padding_n, mat.getPaddingM());
    cl::NDRange local(32, 32);

    GPU::mat_mmul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
         m_padding_n, m_padding_m, mat.getPaddingM(), m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(m_m, m_n);
    cl::NDRange global(m_padding_m, m_padding_n);
    cl::NDRange local(32, 32);

    cl::Buffer block_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 32*33);

    GPU::mat_tranpose_kernel(cl::EnqueueArgs(GPU::queue, global, local),
         m_padding_n, m_padding_m, m_buffer, res.getBuffer(), block_buffer);

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator+(const Matrix &mat) const {
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid addition Mat+Mat");
    }

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_add_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator-(const Matrix &mat) const {
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid substraction Mat-Mat");
    }

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_sub_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator/(const Matrix &mat) const {
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid division Mat/Mat");
    }

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_div_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator+(const double coeff) const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_add_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator/(const double coeff) const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_div_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::hadamardProduct(const Matrix &mat) const {
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid Matrix Hadamard Product");
    }

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::sqrt() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sqrt_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::log() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_log_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer,res.getBuffer());

    GPU::queue.finish();

    return res;
}


VectorGPU Matrix::operator*(const VectorGPU &vec) const {
    if (vec.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Vec");
    }
    VectorGPU res(m_n);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_vmul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::operator+(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid addition Mat+Vec");
    }

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    std::vector<double> vec_copy(vec.getPaddingN());
    cl::copy(GPU::queue, vec.getBuffer(), vec_copy.begin(), vec_copy.end());
    std::vector<double> matrix_cpu(m_padding_n * m_padding_m, 0);
    for (int i = 0; i<m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            matrix_cpu[i * m_padding_m + j] = vec_copy[i];
        }
    }

    cl::Buffer buff = cl::Buffer(GPU::context, matrix_cpu.begin(), matrix_cpu.end(), true);

    GPU::mat_add_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, buff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

double Matrix::sumElem() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    int block_size = 16;
    cl::NDRange local(block_size, block_size);
    int partial_sums_size = m_padding_n/block_size * m_padding_m/block_size;

    cl::Buffer partial_sums = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * partial_sums_size);

    GPU::mat_sum_elem_kernel(cl::EnqueueArgs(GPU::queue, global, local),
        m_padding_n, m_padding_m, m_buffer, partial_sums);

    GPU::queue.finish();
    std::vector<double> mat_copy(partial_sums_size);
    cl::copy(GPU::queue, partial_sums, mat_copy.begin(), mat_copy.end());
    double s = 0;
    for (int i = 0; i<partial_sums_size; i++) {
        s += mat_copy[i];
    }

    return s;
}

double Matrix::computeMetric(const Matrix& y) const {
    Matrix result_matrix(m_n, m_m);
    std::vector<double> mat_copy(m_padding_m * m_padding_n);
    std::vector<double> y_copy(y.getPaddingM() * y.getPaddingN());
    std::vector<double> result_vector(m_padding_m * m_padding_n, 0);
    cl::copy(GPU::queue, m_buffer, mat_copy.begin(), mat_copy.end());
    cl::copy(GPU::queue, y.getBuffer(), y_copy.begin(), y_copy.end());
    //out << "Size ("  << n << " * " << m << ")" << std::endl ;
    for (int j = 0; j < m_m; j++) {
        double max_val = mat_copy[j];
        int max_i = 0;
        result_vector[j] = 1;
        for (int i = 1; i < m_n; i++) {
            if (mat_copy[i * m_padding_m + j] <= max_val) {
                result_vector[i * m_padding_m + j] = 0;
            } else { // for the new max
                result_vector[max_i* m_padding_m + j] = 0; // set old max to 0
                max_i = i;
                max_val = mat_copy[i * m_padding_m + j];
                result_vector[i * m_padding_m + j] = 1; // set new max to 1
            }
        }
    }
    int n_errors = 0;
    for (int j = 0; j < m_m; j++) {
        for (int i = 0; i < m_n; i++) {
             if (result_vector[i * m_padding_m + j] != y_copy[i * m_padding_m + j]) {
                 n_errors++;
                 break;
             }
        }
    }
    return 1 - ((double) n_errors / m_m);
}

void Matrix::fillWithZeros() {
    cl::NDRange global(m_padding_m, m_padding_n);
    cl::NDRange local(32, 32);

    GPU::mat_fill_with_zeros_kernel(cl::EnqueueArgs(GPU::queue, global, local),
         m_padding_n, m_padding_m, m_buffer);

    GPU::queue.finish();
}

void Matrix::fillRandomly() {
    double weights_init = std::sqrt(6.0 / (m_n + m_m));
    std::vector<double> matrix_cpu(m_padding_n * m_padding_m, 0);
    for (int i = 0; i<m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            matrix_cpu[i * m_padding_m + j] = (((double) rand() / (double) RAND_MAX) * 2 * weights_init - weights_init);
        }
    }
    m_buffer = cl::Buffer(GPU::context, matrix_cpu.begin(), matrix_cpu.end(), true);
}

Matrix operator*(const double coeff, const Matrix& mat) {
    Matrix res(mat.getN(), mat.getM());
    cl::NDRange global(mat.getPaddingN(), mat.getPaddingM());
    cl::NDRange local(16, 16);

    GPU::mat_coeff_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             mat.getPaddingN(), mat.getPaddingM(), coeff, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix operator-(const double coeff, const Matrix& mat) {
    Matrix res(mat.getN(), mat.getM());
    cl::NDRange global(mat.getPaddingN(), mat.getPaddingM());
    cl::NDRange local(16, 16);

    GPU::mat_coeff_sub_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             mat.getPaddingN(), mat.getPaddingM(), mat.getN(),  mat.getM(), coeff, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}


Matrix Matrix::computeLinearDev() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_linear_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeSigmoidEval() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sigmoid_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeSigmoidDev() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sigmoid_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeReLUEval() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_relu_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeReLUDev() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_relu_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeTanhEval() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_tanh_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeTanhDev() const {

    Matrix res(m_n, m_m);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_tanh_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

Matrix Matrix::computeSoftmaxEval() const {

    Matrix res(m_n, m_m);
    std::vector<double> mat_copy(m_padding_n * m_padding_m);
    std::vector<double> sum_exp(m_padding_m, 0);
    cl::copy(GPU::queue, m_buffer, mat_copy.begin(), mat_copy.end());
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            sum_exp[j] += std::exp(mat_copy[m_padding_m * i + j]);
        }
    }
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    cl::Buffer buffer = cl::Buffer(GPU::context, sum_exp.begin(), sum_exp.end(), true);

    GPU::mat_softmax_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_n, m_m, buffer, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}


std::ostream& operator << (std::ostream& out, const Matrix& mat) {
    int n = mat.getN(); int m = mat.getM();
    std::vector<double> mat_copy(mat.getPaddingN()*mat.getPaddingM());
    cl::copy(GPU::queue, mat.getBuffer(), mat_copy.begin(), mat_copy.end());
    for (int i = 0; i < mat.getN(); i++) {
        for (int j = 0; j < mat.getM(); j++) {
            out << mat_copy[j + mat.getPaddingM()*i] << " ";
        }
        out << std::endl;
    }
    return out;
}
