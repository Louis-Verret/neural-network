#include "Matrix.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

Matrix::Matrix() : m_n(0), m_m(0) {
}

Matrix::~Matrix() {
}

Matrix::Matrix(int n, int m) :
        m_n(n),
        m_m(m) {
    m_coefficients = std::vector<double>(n * m);
}

const double &Matrix::operator()(int i, int j) const {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    throw std::logic_error("Invalid matrix element");
}

double &Matrix::operator()(int i, int j) {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    throw std::logic_error("Invalid matrix element");
}

Vector Matrix::operator*(const Vector &vec) const {
    if ((int) vec.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Vec");
    }

    Vector vec_s(m_n);
    for (int i = 0; i < m_n; i++) {
        double vec_s_i = 0;
        for (int j = 0; j < m_m; j++) {
            vec_s_i += m_coefficients[i * m_m + j] * vec(j);
        }
        vec_s(i) = vec_s_i;
    }
    return vec_s;
}

Matrix Matrix::operator*(const Matrix &mat) const {
    if (mat.getN() != m_m) {
        // std::cout << mat.getN() << " " << m_m << std::endl;
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    Matrix mat_s(m_n, mat.getM());
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < mat.getM(); j++) {
            double mat_s_ij = 0;
            for (int k = 0; k < m_m; k++) {
                mat_s_ij += m_coefficients[i * m_m + k] * mat(k, j);
            }
            mat_s(i, j) = mat_s_ij;
        }
    }
    return mat_s;
}

Matrix Matrix::operator+(const Vector &vec) const {
    if (m_n != (int) vec.getN()) {
        throw std::logic_error("Invalid addition Mat+Vec");
    }

    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + vec(i);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid substraction Mat-Mat");
    }

    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] - mat(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid addition Mat+Mat");
    }

    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + mat(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator+(const double coeff) const {
    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + coeff;
        }
    }
    return result;
}

Matrix Matrix::operator/(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid division Mat/Mat");
    }

    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] / mat(i,j);
        }
    }
    return result;
}


Matrix Matrix::operator/(const double coeff) const {
    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j]/coeff;
        }
    }
    return result;
}

double Matrix::sumElem() const {
    double sum = 0;
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            sum += m_coefficients[i * m_m + j];
        }
    }
    return sum;
}

Matrix Matrix::hadamardProduct(const Matrix &mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid Matrix Hadamard Product");
    }

    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] * mat(i, j);
        }
    }
    return result;
}

Matrix Matrix::sqrt() const {
    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            if (m_coefficients[i* m_m +j] < 0) {
                throw std::logic_error("Invalid Matrix sqrt");
            }
            result(i, j) = std::sqrt(m_coefficients[i * m_m + j]);
        }
    }
    return result;
}

Matrix Matrix::log() const {
    Matrix result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            if (m_coefficients[i* m_m +j] <= 0) {
                throw std::logic_error("Invalid Matrix log");
            }
            result(i, j) = std::log(m_coefficients[i * m_m + j]);
        }
    }
    return result;
}

Matrix Matrix::argmax() const {
    Matrix result(m_n, m_m);
    for (int j = 0; j < m_m; j++) {
        double max_val = m_coefficients[j];
        double max_i = 0;
        result(0, j) = 1;
        for (int i = 1; i < m_n; i++) {
            if (m_coefficients[i* m_m +j] <= max_val) {
                result(i, j) = 0;
            } else { // for the new max
                result(max_i, j) = 0; // set old max to 0
                max_i = i;
                max_val = m_coefficients[i* m_m +j];
                result(i, j) = 1; // set new max to 1
            }
        }
    }
    return result;
}

Matrix Matrix::generateBitMatrix(int n, int m, double bit_rate) {
    Matrix result(n, m);
    for (int i = 0; i<n; i++) {
        for (int j =0; j<m; j++) {
            double r = ((double) rand() / (double) RAND_MAX);
            if (r < bit_rate) {
                result(i, j) = 1;
            } else {
                result(i, j) = 0;
            }
        }
    }
    return result;
}

void Matrix::fillRandomly() {
    for (int i = 0; i < m_m * m_n; i++) {
        double weights_init = std::sqrt(12 / (m_n + m_m));
        double r = ((double) rand() / (double) RAND_MAX) * 2 * weights_init - weights_init;
        m_coefficients[i] = r;
    }
}

void Matrix::fillWithZero() {
    for (int i = 0; i < m_m * m_n; i++) {
        m_coefficients[i] = 0;
    }
}

Matrix Matrix::transpose() const { //not cache aware
    Matrix transpose(m_m, m_n);
    for (int i = 0; i<m_m; i++) {
        for (int j =0; j<m_n; j++) {
            transpose(i, j) = m_coefficients[j * m_m + i];
        }
    }
    return transpose;
}

void Matrix::resize(int new_n, int new_m) {
    m_coefficients = std::vector<double>(new_n * new_m);
    m_n = new_n;
    m_m = new_m;
}

std::ostream& operator << (std::ostream& out, const Matrix& mat) {
    int n = mat.getN(); int m = mat.getM();
    //out << "Size ("  << n << " * " << m << ")" << std::endl ;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << mat(i,j) << " ";
        }
        out << std::endl;
    }
    return out;
}

Matrix operator*(const double coeff, const Matrix& mat) {
    int n = mat.getN();
    int m = mat.getM();
    Matrix result(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result(i, j) = coeff * mat(i,j);
        }
    }
    return result;
}

Matrix operator-(const double coeff, const Matrix& mat) {
    int n = mat.getN();
    int m = mat.getM();
    Matrix result(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result(i, j) = coeff - mat(i,j);
        }
    }
    return result;
}
