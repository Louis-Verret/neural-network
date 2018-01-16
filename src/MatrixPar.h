#ifndef MATRIXPAR
#define MATRIXPAR

#include <vector>
#include <cstdlib>
#include <ostream>

#include "Vector.h"


class MatrixPar {
public:
    MatrixPar();
    MatrixPar(int n, int m);
    ~MatrixPar();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    double &operator()(int i, int j);
    const double &operator()(int i, int j) const;

    MatrixPar operator*(const MatrixPar &mat) const;
    MatrixPar operator-(const MatrixPar& mat) const;
    MatrixPar operator+(const MatrixPar& mat) const;
    MatrixPar operator+(const Vector &vec) const;
    MatrixPar operator+(const double coeff) const;
    MatrixPar operator/(const double coeff) const;
    MatrixPar operator/(const MatrixPar& mat) const;
    Vector operator*(const Vector &vec) const;

    void fillRandomly();
    void fillWithZero();
    MatrixPar transpose() const;
    double sumElem() const;
    void resize(int new_n, int new_m);
    static MatrixPar generateBitMatrix(int n, int m, double bit_rate);
    MatrixPar sqrt() const;
    MatrixPar log() const;
    MatrixPar argmax() const;
    MatrixPar hadamardProduct(const MatrixPar &mat2) const;

protected:
    int m_n;
    int m_m;
    std::vector<double> m_coefficients;

};

std::ostream& operator << (std::ostream& out, const MatrixPar& m);
MatrixPar operator*(const double coeff, const MatrixPar& mat);
MatrixPar operator-(const double coeff, const MatrixPar& mat);


#endif
