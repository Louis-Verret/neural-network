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
    //Vector(const Vector& vec);
    ~Vector();

    int getN() const { return m_n; };
    double &operator()(int i);
    const double &operator()(int i) const;

    Vector operator+(const Vector &v) const;
    Vector operator-(const Vector &v) const;
    Vector operator*(const Vector &v) const;
    Vector operator/(const Vector &v) const;
    Vector operator+(const double coeff) const;
    Vector operator/(const double coeff) const;

    //Vector& operator=(const Vector& mat);

    void fillRandomly();
    void fillWithZero();
    Vector sqrt() const;

protected:
    int m_n;
    //double* m_coefficients;
    std::vector<double> m_coefficients;
};

std::ostream& operator << (std::ostream& out, const Vector &v);
Vector operator*(const double coeff, const Vector &v);

#endif
