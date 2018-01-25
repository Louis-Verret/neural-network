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
                h_A[m_padding_m*i +j] = i + j;
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

    GPU::mat_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
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
