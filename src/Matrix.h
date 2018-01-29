#ifndef Matrix
#define Matrix

#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#include "../Common/cl.hpp"
#include "../Common/util.hpp"
#include "GPU.h"
#include "VectorGPU.h"

class Matrix {
public:
    Matrix();
    Matrix(int n, int m);
    ~Matrix();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    int getPaddingN() const { return m_padding_n; };
    int getPaddingM() const { return m_padding_m; };
    void setBuffer(cl::Buffer& buffer) {m_buffer = buffer;};
    cl::Buffer getBuffer() const { return m_buffer; };
    const double &operator()(cl::CommandQueue& queue, int i, int j) const;
    Matrix operator*(const Matrix& mat) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix operator/(const Matrix& mat) const;
    Matrix operator+(const VectorGPU &vec) const;
    VectorGPU operator*(const VectorGPU &vec) const;
    Matrix operator+(const double coeff) const;
    Matrix operator/(const double coeff) const;
    Matrix hadamardProduct(const Matrix& mat) const;
    Matrix transpose() const;
    Matrix sqrt() const;
    Matrix log() const;
    double sumElem() const;
    double computeMetric() const;

    void fillWithZeros();
    void fillRandomly();

protected:
    int m_padding_n;
    int m_padding_m;
    int m_n;
    int m_m;
    cl::Buffer m_buffer;
};

std::ostream& operator << (std::ostream& out, const Matrix& mat);
Matrix operator*(const double coeff, const Matrix& mat);
Matrix operator-(const double coeff, const Matrix& mat);


#endif //Matrix
