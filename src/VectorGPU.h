#ifndef VECTOR_GPU
#define VECTOR_GPU

#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"

class VectorGPU {
public:
    VectorGPU();
    VectorGPU(int n, bool init = true);
    VectorGPU(int n, double val);
    ~VectorGPU();

    int getN() const { return m_n; };
    int getPaddingN() const { return m_padding_n; };
    cl::Buffer getBuffer() const { return m_buffer; };
    // double &operator()(int i);
    // const double &operator()(int i) const;

    VectorGPU operator+(const VectorGPU &v) const;
    VectorGPU operator-(const VectorGPU &v) const;
    VectorGPU operator*(const VectorGPU &v) const;
    VectorGPU operator/(const VectorGPU &v) const;
    VectorGPU operator+(const double coeff) const;
    VectorGPU operator/(const double coeff) const;

    void fillRandomly();
    void fillWithZeros();
    VectorGPU sqrt() const;

protected:
    int m_n;
    int m_padding_n;
    cl::Buffer m_buffer;
};

std::ostream& operator << (std::ostream& out, const VectorGPU &v);
VectorGPU operator*(const double coeff, const VectorGPU &v);

#endif
