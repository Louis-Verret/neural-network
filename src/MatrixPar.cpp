#include "MatrixPar.h"
#include <iostream>
#include <cmath>

MatrixPar::MatrixPar() : m_n(0), m_m(0) {
}

MatrixPar::~MatrixPar() {
}

MatrixPar::MatrixPar(int n, int m) :
        m_n(n),
        m_m(m) {
    m_coefficients = std::vector<double>(n * m);
}

const double &MatrixPar::operator()(int i, int j) const {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    throw std::logic_error("Invalid matrix element");
}

double &MatrixPar::operator()(int i, int j) {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    throw std::logic_error("Invalid matrix element");
}

Vector MatrixPar::operator*(const Vector &vec) const {
    if (vec.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Vec");
    }

    Vector vec_s(m_n);
    int j = 0; double vec_s_i = 0;
    #pragma omp parallel shared(vec, vec_s) private(j, vec_s_i)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            vec_s_i = 0;
            for (j = 0; j < m_m; j++) {
                vec_s_i += m_coefficients[i * m_m + j] * vec(j);
            }
            vec_s(i) = vec_s_i;
        }
    }
    return vec_s;
}

MatrixPar MatrixPar::operator*(const MatrixPar &mat) const {
    if (mat.getN() != m_m) {
        // std::cout << mat.getN() << " " << m_m << std::endl;
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    int m = mat.getM();
    MatrixPar mat_s(m_n, mat.getM());
    int j = 0; int k = 0; double mat_s_ij = 0;
    #pragma omp parallel shared(mat, mat_s) private(j, k, mat_s_ij)
    {
        #pragma omp for
        for (int i = 0; i < m_n; i++) {
            for (j = 0; j < m; j++) {
                mat_s_ij = 0;
                for (k = 0; k < m_m; k++) {
                    mat_s_ij += m_coefficients[i * m_m + k] * mat(k, j);
                }
                mat_s(i, j) = mat_s_ij;
            }
        }
    };
    return mat_s;
}

MatrixPar MatrixPar::operator+(const Vector &vec) const {
    if (m_n != (int) vec.getN()) {
        throw std::logic_error("Invalid addition Mat+Vec");
    }

    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + vec(i);
        }
    }
    return result;
}

MatrixPar MatrixPar::operator-(const MatrixPar& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid substraction Mat-Mat");
    }

    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] - mat(i,j);
        }
    }
    return result;
}

MatrixPar MatrixPar::operator+(const MatrixPar& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid addition Mat+Mat");
    }

    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + mat(i,j);
        }
    }
    return result;
}

MatrixPar MatrixPar::operator+(const double coeff) const {
    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] + coeff;
        }
    }
    return result;
}

MatrixPar MatrixPar::operator/(const MatrixPar& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid division Mat/Mat");
    }

    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] / mat(i,j);
        }
    }
    return result;
}


MatrixPar MatrixPar::operator/(const double coeff) const {
    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j]/coeff;
        }
    }
    return result;
}

double MatrixPar::sumElem() const {
    double sum = 0;
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            sum += m_coefficients[i * m_m + j];
        }
    }
    return sum;
}

MatrixPar MatrixPar::hadamardProduct(const MatrixPar &mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid Matrix Hadamard Product");
    }

    MatrixPar result(m_n, m_m);
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            result(i, j) = m_coefficients[i * m_m + j] * mat(i, j);
        }
    }
    return result;
}

MatrixPar MatrixPar::sqrt() const {
    MatrixPar result(m_n, m_m);
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

MatrixPar MatrixPar::log() const {
    MatrixPar result(m_n, m_m);
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

MatrixPar MatrixPar::argmax() const {
    MatrixPar result(m_n, m_m);
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

MatrixPar MatrixPar::generateBitMatrix(int n, int m, double bit_rate) {
    MatrixPar result(n, m);
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

void MatrixPar::fillRandomly() {
    for (int i = 0; i < m_m * m_n; i++) {
        double weights_init = 4 * std::sqrt(6 / (m_n + m_m));
        double r = ((double) rand() / (double) RAND_MAX) * 2 * weights_init - weights_init;
        m_coefficients[i] = r;
    }
}

void MatrixPar::fillWithZero() {
    for (int i = 0; i < m_m * m_n; i++) {
        m_coefficients[i] = 0;
    }
}

MatrixPar MatrixPar::transpose() const { //not cache aware
    MatrixPar transpose(m_m, m_n);
    for (int i = 0; i<m_m; i++) {
        for (int j =0; j<m_n; j++) {
            transpose(i, j) = m_coefficients[j * m_m + i];
        }
    }
    return transpose;
}

void MatrixPar::resize(int new_n, int new_m) {
    m_coefficients = std::vector<double>(new_n * new_m);
    m_n = new_n;
    m_m = new_m;
}

std::ostream& operator << (std::ostream& out, const MatrixPar& mat) {
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

MatrixPar operator*(const double coeff, const MatrixPar& mat) {
    int n = mat.getN();
    int m = mat.getM();
    MatrixPar result(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result(i, j) = coeff * mat(i,j);
        }
    }
    return result;
}

MatrixPar operator-(const double coeff, const MatrixPar& mat) {
    int n = mat.getN();
    int m = mat.getM();
    MatrixPar result(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result(i, j) = coeff - mat(i,j);
        }
    }
    return result;
}
