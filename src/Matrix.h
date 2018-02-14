#ifndef MATRIX
#define MATRIX

#include <vector>
#include <cstdlib>
#include <ostream>

#include "Vector.h"

/** Class that implements a matrix container optimized
    with parallel computations using OpenMP */

class Matrix {
public:

    /* Constructors / Destructor */
    Matrix();
    Matrix(int n, int m);
    Matrix(const Matrix& mat);
    ~Matrix();

    /* Get Methods */
    int getN() const { return m_n; };
    int getM() const { return m_m; };

    /* Matrix operators */
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
    Matrix& operator=(const Matrix& mat);

    /* Init methods */
    void fillRandomly();
    void fillWithZero();

    /* Mathematical methods */
    Matrix transpose() const;
    double sumElem() const;
    Matrix sqrt() const;
    Matrix log() const;
    Matrix argmax() const;
    Matrix hadamardProduct(const Matrix &mat2) const;

    /* Miscellaneous methods */
    void resize(int new_n, int new_m);
    static Matrix generateBitMatrix(int n, int m, double bit_rate);

protected:
    int m_n;
    int m_m;
    double* m_coefficients;

};

/* Extern operators */
std::ostream& operator << (std::ostream& out, const Matrix& m);
Matrix operator*(const double coeff, const Matrix& mat);
Matrix operator-(const double coeff, const Matrix& mat);


#endif
