#ifndef MATRIXGPU
#define MATRIXGPU

#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"

class MatrixGPU {
public:
    MatrixGPU();
    MatrixGPU(int n, int m, bool init = true);
    ~MatrixGPU();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    int getPaddingN() const { return m_padding_n; };
    int getPaddingM() const { return m_padding_m; };
    cl::Buffer getBuffer() const { return m_buffer; };
    const double &operator()(cl::CommandQueue& queue, int i, int j) const;
    MatrixGPU operator*(const MatrixGPU &mat) const;
    MatrixGPU operator+(const MatrixGPU& mat) const;
    MatrixGPU transpose() const;
    double sumElem() const;

    void fillWithZeros();
    void fillRandomly();

protected:
    int m_padding_n;
    int m_padding_m;
    int m_n;
    int m_m;
    cl::Buffer m_buffer;
};

std::ostream& operator << (std::ostream& out, const MatrixGPU& mat);


#endif //MATRIXGPU
