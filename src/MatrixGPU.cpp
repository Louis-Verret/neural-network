#include <stdexcept>
#include "MatrixGPU.h"

MatrixGPU::MatrixGPU() : m_n(0), m_m(0) {
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 0 * 0);
}

MatrixGPU::~MatrixGPU() {
}

MatrixGPU::MatrixGPU(int n, int m, bool init) : m_n(n), m_m(m) {
    int block_size = 32;
    int padding_n = (m_n%block_size != 0)? m_n + block_size - m_n%block_size : m_n;
    int padding_m = (m_m%block_size != 0)? m_m + block_size - m_m%block_size : m_m;
    m_padding_n = padding_n;
    m_padding_m = padding_m;
    if (init) {
        std::vector<double> h_A(m_padding_n * m_padding_m, 0);
        for (int i = 0; i<m_n; i++) {
            for (int j = 0; j<m_m; j++) {
                h_A[m_padding_m*i +j] = (int)(((double) rand() / (double) RAND_MAX) * 10);
            }
        }
        // std::cout << h_A[31 + m_m * 0] << std::endl;
        // std::cout << h_A[30 + m_m * 0] << std::endl;
        // std::cout << "------" << std::endl;
        // for (int i = 0; i<m_padding_n; i++) {
        //     for (int j = 0; j<m_padding_m; j++) {
        //         if (h_A[i + m_padding_m*j] != 0.0) {
        //             std::cout << i << " " << j << " " << h_A[i + m_padding_m*j] << std::endl;
        //         }
        //     }
        // }
        // std::cout << "------" << std::endl;
        m_buffer = cl::Buffer(GPU::context, h_A.begin(), h_A.end(), true);
    } else {
        m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * m_padding_n * m_padding_m);
    }
}

// const double &MatrixGPU::operator()(cl::CommandQueue& queue, int i, int j) const {
//     if (i < m_n && j < m_m) {
//         std::vector<float> v(m_n * m_m);
//         cl::copy(queue, m_buff, v.begin(), v.end());
//         return v[i * m_m + j];
//     }
//     throw std::logic_error("Invalid matrix element");
// }


MatrixGPU MatrixGPU::operator*(const MatrixGPU &mat) const {

    if (mat.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    MatrixGPU res(m_n, mat.getM(), false);
    // std::cout << m_padding_n << std::endl;
    // std::cout << m_padding_m << std::endl;
    // std::cout << mat.getPaddingM() << std::endl;
    // std::cout << mat.getPaddingN() << std::endl;
    // std::cout << res.getPaddingN() << std::endl;
    // std::cout << res.getPaddingM() << std::endl;
    cl::NDRange global(m_padding_n, mat.getPaddingM());
    cl::NDRange local(32, 32);

    GPU::mat_mmul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
         m_padding_n, m_padding_m, mat.getPaddingM(), m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    // std::vector<float> h_C(m_padding_n*mat.getPaddingM());
    //
    // cl::copy(GPU::queue, res.getBuff(), h_C.begin(), h_C.end());

    //std::cout << "res:" << std::endl;
    // std::cout << "------" << std::endl;
    // for (int i = 0; i<m_padding_n; i++) {
    //     for (int j = 0; j<mat.getPaddingM(); j++) {
    //         if (h_C[i + mat.getPaddingM()*j] != 0) {
    //             std::cout << i << " " << j << " " << h_C[i + mat.getPaddingM()*j] << std::endl;
    //         }
    //     }
    // }
    // std::cout << "------" << std::endl;

    return res;
}

MatrixGPU MatrixGPU::transpose() const {
    MatrixGPU res(m_m, m_n, false);
    cl::NDRange global(m_padding_m, m_padding_n);
    cl::NDRange local(32, 32);

    cl::Buffer block_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 32*33);

    GPU::mat_tranpose_kernel(cl::EnqueueArgs(GPU::queue, global, local),
         m_padding_n, m_padding_m, m_buffer, res.getBuffer(), block_buffer);

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::operator+(const MatrixGPU &mat) const {
    //std::cout << mat.getN() << " " << m_n << " " << mat.getM() << " " << m_m << std::endl;
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid addition Mat+Mat");
    }

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_add_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::operator-(const MatrixGPU &mat) const {
    //std::cout << mat.getN() << " " << m_n << " " << mat.getM() << " " << m_m << std::endl;
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid substraction Mat-Mat");
    }

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_sub_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::operator/(const MatrixGPU &mat) const {
    //std::cout << mat.getN() << " " << m_n << " " << mat.getM() << " " << m_m << std::endl;
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid division Mat/Mat");
    }

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_div_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::operator+(const double coeff) const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_add_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::operator/(const double coeff) const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_div_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::hadamardProduct(const MatrixGPU &mat) const {
    //std::cout << mat.getN() << " " << m_n << " " << mat.getM() << " " << m_m << std::endl;
    if (mat.getN() != m_n || mat.getM() != m_m) {
        throw std::logic_error("Invalid Matrix Hadamard Product");
    }

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(32, 32);

    GPU::mat_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::sqrt() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sqrt_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::log() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_log_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer,res.getBuffer());

    GPU::queue.finish();

    return res;
}


VectorGPU MatrixGPU::operator*(const VectorGPU &vec) const {
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

MatrixGPU MatrixGPU::operator+(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid addition Mat+Vec");
    }
    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_add_vec_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

double MatrixGPU::sumElem() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    int block_size = 32;
    cl::NDRange local(block_size, block_size);
    int partial_sums_size = m_padding_n/block_size * m_padding_m/block_size;

    cl::Buffer partial_sums = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * partial_sums_size);
    // std::vector<double> h_A(m_padding_n * m_padding_m, 0);
    // cl::Buffer localSums = cl::Buffer(GPU::context, h_A.begin(), h_A.end(), true);

    GPU::mat_sum_elem_kernel(cl::EnqueueArgs(GPU::queue, global, local),
        m_padding_n, m_padding_m, m_buffer, partial_sums);

    GPU::queue.finish();
    std::vector<double> mat_copy(partial_sums_size);
    cl::copy(GPU::queue, partial_sums, mat_copy.begin(), mat_copy.end());
    double s = 0;
    for (int i = 0; i<partial_sums_size; i++) {
        //std::cout << mat_copy[i] << std::endl;
        s += mat_copy[i];
    }

    return s;
}

MatrixGPU MatrixGPU::computeLinearDev() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_linear_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeSigmoidEval() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sigmoid_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeSigmoidDev() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_sigmoid_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeReLUEval() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_relu_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeReLUDev() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_relu_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeTanhEval() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_tanh_eval_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeTanhDev() const {

    MatrixGPU res(m_n, m_m, false);
    cl::NDRange global(m_padding_n, m_padding_m);
    cl::NDRange local(16, 16);

    GPU::mat_tanh_dev_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_padding_m, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU MatrixGPU::computeSoftmaxEval() const {

    MatrixGPU res(m_n, m_m, false);
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
                             m_padding_n, m_padding_m, buffer, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU operator*(const double coeff, const MatrixGPU& mat) {
    MatrixGPU res(mat.getN(), mat.getM(), false);
    cl::NDRange global(mat.getPaddingN(), mat.getPaddingM());
    cl::NDRange local(16, 16);

    GPU::mat_coeff_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             mat.getPaddingN(), mat.getPaddingM(), coeff, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

MatrixGPU operator-(const double coeff, const MatrixGPU& mat) {
    MatrixGPU res(mat.getN(), mat.getM(), false);
    cl::NDRange global(mat.getPaddingN(), mat.getPaddingM());
    cl::NDRange local(16, 16);

    GPU::mat_coeff_sub_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             mat.getPaddingN(), mat.getPaddingM(), coeff, mat.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

std::ostream& operator << (std::ostream& out, const MatrixGPU& mat) {
    int n = mat.getN(); int m = mat.getM();
    std::vector<double> mat_copy(mat.getPaddingN()*mat.getPaddingM());
    cl::copy(GPU::queue, mat.getBuffer(), mat_copy.begin(), mat_copy.end());
    //out << "Size ("  << n << " * " << m << ")" << std::endl ;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << mat_copy[j + mat.getPaddingM()*i] << " ";
        }
        out << std::endl;
    }
    return out;
}
