#ifndef VECTOR_GPU
#define VECTOR_GPU

#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"

/** Class that implements a vector container optimized
    with GPU computations using OpenCL (Similar to Matrix.h)*/

class VectorGPU {
public:

    /* Constructors / Destructor */
    VectorGPU();
    VectorGPU(int n, bool init = true);
    VectorGPU(int n, double val);
    ~VectorGPU();

    /* Get Methods */
    int getN() const { return m_n; };
    int getPaddingN() const { return m_padding_n; };
    cl::Buffer getBuffer() const { return m_buffer; };

    /* Operators */
    VectorGPU operator+(const VectorGPU &v) const;
    VectorGPU operator-(const VectorGPU &v) const;
    VectorGPU operator*(const VectorGPU &v) const;
    VectorGPU operator/(const VectorGPU &v) const;
    VectorGPU operator+(const double coeff) const;
    VectorGPU operator/(const double coeff) const;

    /* Init methods */
    void fillRandomly();
    void fillWithZeros();

    /* Mathematical method */
    VectorGPU sqrt() const;

protected:
    int m_n;
    int m_padding_n;
    cl::Buffer m_buffer;
};

/* Extern method */
std::ostream& operator << (std::ostream& out, const VectorGPU &v);
VectorGPU operator*(const double coeff, const VectorGPU &v);

#endif
