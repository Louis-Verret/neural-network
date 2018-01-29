#include <stdexcept>
#include "VectorGPU.h"

VectorGPU::VectorGPU() : m_n(0) {
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * 0);
}

VectorGPU::~VectorGPU() {
}

VectorGPU::VectorGPU(int n, bool init) : m_n(n) {
    int block_size = 32;
    int padding_n = (m_n%block_size != 0)? m_n + block_size - m_n%block_size : m_n;
    m_padding_n = padding_n;
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * m_padding_n);
}

VectorGPU::VectorGPU(int n, double val) : m_n(n) {
    int block_size = 32;
    int padding_n = (m_n%block_size != 0)? m_n + block_size - m_n%block_size : m_n;
    m_padding_n = padding_n;
    m_buffer = cl::Buffer(GPU::context, CL_MEM_READ_WRITE, sizeof(double) * m_padding_n);

    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_fill_with_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, val);

    GPU::queue.finish();
}

VectorGPU VectorGPU::operator+(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid addition Vec+Vec");
    }

    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_add_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::operator-(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid substraction Vec-Vec");
    }

    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_sub_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::operator*(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid multiplication Vec*Vec");
    }

    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::operator/(const VectorGPU &vec) const {
    if (vec.getN() != m_n) {
        throw std::logic_error("Invalid division Vec/Vec");
    }

    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_div_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, vec.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::operator+(const double coeff) const {
    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_add_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::operator/(const double coeff) const {
    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_div_coeff_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, coeff, res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU VectorGPU::sqrt() const {
    VectorGPU res(m_n, false);
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_sqrt_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer, res.getBuffer());

    GPU::queue.finish();

    return res;
}

VectorGPU operator*(const double coeff, const VectorGPU &v) {
    VectorGPU res(v.getN(), false);
    cl::NDRange global(v.getPaddingN());
    cl::NDRange local(32);

    GPU::vec_coeff_mul_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             v.getPaddingN(), coeff, v.getBuffer(), res.getBuffer());

    GPU::queue.finish();

    return res;
}

void VectorGPU::fillRandomly() {
    srand(time(NULL));
    std::vector<double> vec_cpu(m_padding_n, 0);
    for (int i = 0; i<m_n; i++) {
        vec_cpu[i] = (int)(((double) rand() / (double) RAND_MAX) * 10 - 5);
    }
    m_buffer = cl::Buffer(GPU::context, vec_cpu.begin(), vec_cpu.end(), true);
}

void VectorGPU::fillWithZeros() {
    cl::NDRange global(m_padding_n);
    cl::NDRange local(32);

    GPU::vec_fill_with_zeros_kernel(cl::EnqueueArgs(GPU::queue, global, local),
                             m_padding_n, m_buffer);

    GPU::queue.finish();
}

std::ostream& operator << (std::ostream& out, const VectorGPU& vec) {
    int n = vec.getN();
    std::vector<double> vec_copy(vec.getPaddingN());
    cl::copy(GPU::queue, vec.getBuffer(), vec_copy.begin(), vec_copy.end());
    for (int i = 0; i < n; i++) {
        out << vec_copy[i] << " ";
    }
    out << std::endl;
    return out;
}
