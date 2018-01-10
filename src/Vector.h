#ifndef VECTOR
#define VECTOR

#include <vector>
#include <cstdlib>
#include <ostream>

class Vector {
public:
    Vector();
    Vector(int n);
    Vector(int n, double val);
    ~Vector();

    int getN() const { return m_n; };
    double &operator()(int i);
    const double &operator()(int i) const;

    Vector operator+(const Vector &v);
    Vector operator-(const Vector &v);
    Vector operator*(const Vector &v);
    Vector operator/(const Vector &v);
    Vector operator+(const double coeff);
    Vector operator/(const double coeff);


    void fillRandomly();
    void fillWithZero();
    Vector sqrt() const;

protected:
    int m_n;
    std::vector<double> m_coefficients;
};

std::ostream& operator << (std::ostream& out, const Vector &v);
Vector operator*(const double coeff, const Vector &v);

#endif
