#ifndef VECTOR
#define VECTOR

#include <vector>
#include <cstdlib>
#include <ostream>

/** Class that implements a vector container optimized
    with parallel computations using OpenMP (Similar to Matrix.h)*/

class Vector {
public:

    /* Constructors / Destructor */
    Vector();
    Vector(int n);
    Vector(int n, double val);
    ~Vector();

    /* Get Methods */
    int getN() const { return m_n; };

    /* Operators */
    double &operator()(int i);
    const double &operator()(int i) const;
    Vector operator+(const Vector &v);
    Vector operator-(const Vector &v);
    Vector operator*(const Vector &v);
    Vector operator/(const Vector &v);
    Vector operator+(const double coeff);
    Vector operator/(const double coeff);

    /* Init methods */
    void fillRandomly();
    void fillWithZero();

    /* Mathematical method */
    Vector sqrt() const;

protected:
    int m_n;
    std::vector<double> m_coefficients;
};

/* Extern methods */
std::ostream& operator << (std::ostream& out, const Vector &v);
Vector operator*(const double coeff, const Vector &v);

#endif
