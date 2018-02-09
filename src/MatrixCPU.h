#ifndef MATRIXCPU_H
#define MATRIXCPU_H

#include <vector>
#include <cstdlib>
#include <ostream>

#include "Vector.h"


class MatrixCPU {
public:
    MatrixCPU();
    MatrixCPU(int n, int m);
    MatrixCPU(const MatrixCPU& mat);
    ~MatrixCPU();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    double &operator()(int i, int j);
    const double &operator()(int i, int j) const;

    MatrixCPU operator*(const MatrixCPU &mat) const;
    MatrixCPU operator-(const MatrixCPU& mat) const;
    MatrixCPU operator+(const MatrixCPU& mat) const;
    MatrixCPU operator+(const Vector &vec) const;
    MatrixCPU operator+(const double coeff) const;
    MatrixCPU operator/(const double coeff) const;
    MatrixCPU operator/(const MatrixCPU& mat) const;
    Vector operator*(const Vector &vec) const;

    MatrixCPU& operator=(const MatrixCPU& mat);

    void fillRandomly();
    void fillWithZeros();
    MatrixCPU transpose() const;
    double sumElem() const;
    void resize(int new_n, int new_m);
    static MatrixCPU generateBitMatrixCPU(int n, int m, double bit_rate);
    MatrixCPU sqrt() const;
    MatrixCPU log() const;
    MatrixCPU argmax() const;
    MatrixCPU hadamardProduct(const MatrixCPU &mat2) const;

protected:
    int m_n;
    int m_m;
    double* m_coefficients;
    //std::vector<double> m_coefficients;

};

std::ostream& operator << (std::ostream& out, const MatrixCPU& m);
MatrixCPU operator*(const double coeff, const MatrixCPU& mat);
MatrixCPU operator-(const double coeff, const MatrixCPU& mat);


#endif
