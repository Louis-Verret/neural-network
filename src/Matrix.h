#ifndef MATRIX
#define MATRIX

#include <vector>
#include <cstdlib>
#include <ostream>

#include "Vector.h"


class Matrix {
public:
    Matrix();
    Matrix(int n, int m);
    ~Matrix();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    double &operator()(int i, int j);
    const double &operator()(int i, int j) const;

    Matrix operator*(const Matrix &mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator+(const Vector &vec) const;
    Matrix operator+(const double coeff) const;
    Matrix operator/(const double coeff) const;
    Matrix operator/(const Matrix& mat) const;
    Vector operator*(const Vector &vec) const;

    void fillRandomly();
    void fillWithZero();
    Matrix transpose() const;
    double sumElem() const;
    void resize(int new_n, int new_m);
    static Matrix generateBitMatrix(int n, int m, double bit_rate);
    Matrix sqrt() const;
    Matrix log() const;
    Matrix argmax() const;
    Matrix hadamardProduct(const Matrix &mat2) const;

protected:
    int m_n;
    int m_m;
    std::vector<double> m_coefficients;

};

std::ostream& operator << (std::ostream& out, const Matrix& m);
Matrix operator*(const double coeff, const Matrix& mat);
Matrix operator-(const double coeff, const Matrix& mat);


#endif
