#include <stdexcept>
#include "MatrixGPU.h"


MatrixGPU::MatrixGPU() : m_n(0), m_m(0) {
    m_context = cl::Context(DEVICE);
    m_buff = cl::Buffer(m_context, CL_MEM_READ_WRITE, sizeof(double) * 0 * 0);
}

MatrixGPU::~MatrixGPU() {
}

MatrixGPU::MatrixGPU(cl::Context& context, int n, int m) : m_n(n), m_m(m) {
    m_context = context;
    m_buff = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * m);
}

const double &MatrixGPU::operator()(cl::CommandQueue& queue, int i, int j) const {
    if (i < m_n && j < m_m) {
        std::vector<float> v(m_n * m_m);
        cl::copy(queue, m_buff, v.begin(), v.end());
        return v[i * m_m + j];
    }
    throw std::logic_error("Invalid matrix element");
}


MatrixGPU MatrixGPU::matmult(cl::CommandQueue& queue, cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> kernel,
                             const MatrixGPU &mat) const {
    if (mat.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    MatrixGPU res(m_context, m_n, m_m);
    cl::NDRange global(m_n, m_m);
    cl::NDRange local(32, 32);
    kernel(cl::EnqueueArgs(queue, global),
               m_n, m_m, m_buff, mat.getBuff(), res.m_buff);
    queue.finish();

    return res;
}


