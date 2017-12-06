#ifndef MATRIX
#define MATRIX

#include <vector>
#include <cstdlib>
#include <ostream>

class Matrix {
public:
    Matrix();
    Matrix(int n, int m);
    ~Matrix();

    int getN() const { return m_n; };
    int getM() const { return m_m; };
    double &operator()(int i, int j);
    const double &operator()(int i, int j) const;

    std::vector<double> operator*(const std::vector<double> &vec) const;
    Matrix transpose() const;

protected:
    const int m_n;
    const int m_m;
    std::vector<double> m_coefficients;

};

std::ostream& operator << (std::ostream& out, const Matrix& m);

#endif
