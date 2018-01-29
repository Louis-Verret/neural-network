#ifndef MATRIXGPU
#define MATRIXGPU

#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"
#include <cmath>
#include "VectorGPU.h"

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
    MatrixGPU operator*(const MatrixGPU& mat) const;
    MatrixGPU operator+(const MatrixGPU& mat) const;
    MatrixGPU operator-(const MatrixGPU& mat) const;
    MatrixGPU operator/(const MatrixGPU& mat) const;
    MatrixGPU operator+(const VectorGPU &vec) const;
    VectorGPU operator*(const VectorGPU &vec) const;
    MatrixGPU operator+(const double coeff) const;
    MatrixGPU operator/(const double coeff) const;
    MatrixGPU hadamardProduct(const MatrixGPU& mat) const;
    MatrixGPU transpose() const;
    MatrixGPU sqrt() const;
    MatrixGPU log() const;
    double sumElem() const;

    MatrixGPU computeLinearDev() const;
    MatrixGPU computeSigmoidEval() const;
    MatrixGPU computeSigmoidDev() const;
    MatrixGPU computeReLUEval() const;
    MatrixGPU computeReLUDev() const;
    MatrixGPU computeTanhEval() const;
    MatrixGPU computeTanhDev() const;
    MatrixGPU computeSoftmaxEval() const;

protected:
    int m_padding_n;
    int m_padding_m;
    int m_n;
    int m_m;
    cl::Buffer m_buffer;
};

std::ostream& operator << (std::ostream& out, const MatrixGPU& mat);
MatrixGPU operator*(const double coeff, const MatrixGPU& mat);
MatrixGPU operator-(const double coeff, const MatrixGPU& mat);


#endif //MATRIXGPU
