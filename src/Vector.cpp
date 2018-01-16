#include "Vector.h"
#include <iostream>
#include <cmath>

Vector::Vector() : m_n(0) {
}

Vector::~Vector() {
}

Vector::Vector(int n) :
        m_n(n)
{
    m_coefficients = std::vector<double>(n);
}

Vector::Vector(int n, double val) :
        m_n(n)
{
    m_coefficients = std::vector<double>(n, val);
}

const double &Vector::operator()(int i) const {
    if (i < m_n)
        return m_coefficients[i];
    throw std::logic_error("Invalid vector element");
}

double &Vector::operator()(int i) {
    if (i < m_n)
        return m_coefficients[i];
    throw std::logic_error("Invalid vector element");
}

Vector Vector::operator+(const Vector &v) {
    if (v.getN() != m_n) {
        std::cout << v.getN() << " " << m_n << std::endl;
        perror("Invalid size for vector addition");
    }
    Vector res(m_n);
    #pragma omp parallel shared(res, v)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            res(i) = m_coefficients[i] + v(i);
        }
    };
    return res;
}

Vector Vector::operator-(const Vector &v) {
    if (v.getN() != m_n) {
        perror("Invalid size for vector substraction");
    }
    Vector res(m_n);
    #pragma omp parallel shared(res, v)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            res(i) = m_coefficients[i] - v(i);
        }
    };
    return res;
}

Vector Vector::operator*(const Vector &v) {
    if (v.getN() != m_n) {
        perror("Invalid size for Hadamard vector product");
    }
    Vector res(m_n);
    #pragma omp parallel shared(res, v)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            res(i) = m_coefficients[i] * v(i);
        }
    };
    return res;
}

Vector Vector::operator/(const Vector &v) {
    if (v.getN() != m_n) {
        perror("Invalid size for vector division");
    }
    Vector res(m_n);
    #pragma omp parallel shared(res, v)
    {
        #pragma omp for
        for (int i = 0; i < v.getN(); i++) {
            res(i) = m_coefficients[i] / v(i);
        }
    };
    return res;
}

Vector Vector::operator/(const double coeff) {
    Vector res(m_n);
    #pragma omp parallel shared(res)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            res(i) = m_coefficients[i] / coeff;
        }
    };
    return res;
}

Vector Vector::operator+(const double coeff) {
    Vector res(m_n);
    #pragma omp parallel shared(res)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            res(i) = m_coefficients[i] + coeff;
        }
    };
    return res;
}

void Vector::fillRandomly() {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            double r = ((double) rand() / (double) RAND_MAX);
            m_coefficients[i] = r;
        }
    };
}

void Vector::fillWithZero() {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            m_coefficients[i] = 0;
        }
    }
}

Vector Vector::sqrt() const {
    Vector res(m_n);
    #pragma omp parallel shared(res)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            if (m_coefficients[i] < 0) {
                throw std::logic_error("Invalid Vector sqrt");
            }
            res(i) = std::sqrt(m_coefficients[i]);
        }
    };
    return res;
}

Vector operator*(const double coeff, const Vector& v) {
    Vector res(v.getN());
    #pragma omp parallel shared(res, v)
    {
        #pragma omp for
        for (int i = 0; i < v.getN(); i++) {
            res(i) = coeff * v(i);
        }
    };
    return res;
}

std::ostream& operator << (std::ostream& out, const Vector& v) {
    int n = v.getN();
    for (int i = 0; i < n; i++) {
        out << v(i) << " ";
    }
    return out;
}
