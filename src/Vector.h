#ifndef VECTOR
#define VECTOR

#include <vector>
#include <cstdlib>
#include <ostream>

/** Class that implements a Vector container without using GPU computations
    This class only serves as a tool for comparing results and execution times
    with and without GPUs **/

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
    Vector operator+(const Vector &v) const;
    Vector operator-(const Vector &v) const;
    Vector operator*(const Vector &v) const;
    Vector operator/(const Vector &v) const;
    Vector operator+(const double coeff) const;
    Vector operator/(const double coeff) const;

    /* Init methods */
    void fillRandomly();
    void fillWithZeros();

    /* Mathematical method */
    Vector sqrt() const;

protected:
    int m_n;
    std::vector<double> m_coefficients;
};

std::ostream& operator << (std::ostream& out, const Vector &v);
Vector operator*(const double coeff, const Vector &v);

#endif
