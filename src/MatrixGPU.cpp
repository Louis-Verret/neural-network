#include <stdexcept>
#include "MatrixGPU.h"

MatrixGPU::MatrixGPU() : m_n(0), m_m(0) {
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 0 * 0);
}

MatrixGPU::~MatrixGPU() {
}

MatrixGPU::MatrixGPU(int n, int m, bool init) : m_n(n), m_m(m) {
    int padding_n = (m_n%32 != 0)? m_n + 32 - m_n%32 : m_n;
    int padding_m = (m_m%32 != 0)? m_m + 32 - m_m%32 : m_m;
    m_padding_n = padding_n;
    m_padding_m = padding_m;
    if (init) {
        std::vector<float> h_A(m_padding_n * m_padding_m, 0);
        for (int i = 0; i<m_n; i++) {
            for (int j = 0; j<m_m; j++) {
                h_A[i + m_padding_m*j] = 1.0;
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
        m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(float) * m_padding_n * m_padding_m);
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

std::ostream& operator << (std::ostream& out, const MatrixGPU& mat) {
    int n = mat.getN(); int m = mat.getM();
    std::vector<float> mat_copy(mat.getPaddingN()*mat.getPaddingM());
    cl::copy(GPU::queue, mat.getBuffer(), mat_copy.begin(), mat_copy.end());
    //out << "Size ("  << n << " * " << m << ")" << std::endl ;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << mat_copy[i + mat.getPaddingM()*j] << " ";
        }
        out << std::endl;
    }
    return out;
}
