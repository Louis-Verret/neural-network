#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"
#include "VectorGPU.h"

/** Class that implements a matrix container optimized
    with GPU computations using OpenCL */

class Matrix {
public:

    /** Constructors */
    Matrix();
    Matrix(int n, int m);
    ~Matrix();

    /** Get Methods */
    int getN() const { return m_n; };
    int getM() const { return m_m; };
    int getPaddingN() const { return m_padding_n; };
    int getPaddingM() const { return m_padding_m; };
    cl::Buffer getBuffer() const { return m_buffer; };

    /* Set Methods */
    void setBuffer(cl::Buffer& buffer) {m_buffer = buffer;};

    /* Matrix operators */
    Matrix operator*(const Matrix& mat) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix operator/(const Matrix& mat) const;
    Matrix operator+(const VectorGPU &vec) const;
    VectorGPU operator*(const VectorGPU &vec) const;
    Matrix operator+(const double coeff) const;
    Matrix operator/(const double coeff) const;
    Matrix hadamardProduct(const Matrix& mat) const; //element-wise multiplication

    /* Mathematical methods */
    Matrix transpose() const;
    Matrix sqrt() const;
    Matrix log() const;
    double sumElem() const;

    /* Init methods */
    void fillWithZeros();
    void fillRandomly();

    /* Neural Net specific methods */
    double computeMetric(const Matrix& y) const;
    Matrix computeLinearDev() const;
    Matrix computeSigmoidEval() const;
    Matrix computeSigmoidDev() const;
    Matrix computeReLUEval() const;
    Matrix computeReLUDev() const;
    Matrix computeTanhEval() const;
    Matrix computeTanhDev() const;
    Matrix computeSoftmaxEval() const;

protected:
    int m_padding_n;
    int m_padding_m;
    int m_n;
    int m_m;
    cl::Buffer m_buffer;
};

/* Extern operators */
std::ostream& operator << (std::ostream& out, const Matrix& mat);
Matrix operator*(const double coeff, const Matrix& mat);
Matrix operator-(const double coeff, const Matrix& mat);


#endif //Matrix
