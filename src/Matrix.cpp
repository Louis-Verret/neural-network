#include "Matrix.h"
#include <iostream>
#include <cmath>
#include "omp.h"
#include <stdexcept>

Matrix::Matrix() : m_n(0), m_m(0) {
    m_coefficients = new double[0];
}

Matrix::~Matrix() {
    delete[] m_coefficients;
}

Matrix::Matrix(int n, int m) :
        m_n(n),
        m_m(m) {
    //m_coefficients = std::vector<double>(n * m);
    m_coefficients = new double[n * m];
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

Matrix Matrix::operator*(const Matrix &mat) const {
    if (mat.getN() != m_m) {
        throw std::logic_error("Invalid multiplication Mat*Mat");
    }

    int m = mat.getM();
    Matrix mat_s(m_n, mat.getM());
    int j = 0; int k = 0; double mat_s_ij = 0;
    Matrix tm = mat.transpose();
    #pragma omp parallel shared(tm, mat_s) private(j, k, mat_s_ij)
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < m_n; i++) {
            for (j = 0; j < m; j++) {
                mat_s_ij = 0;
                for (k = 0; k < m_m; k++) {
                    mat_s_ij += m_coefficients[i * m_m + k] * tm(j, k);
                }
                mat_s(i, j) = mat_s_ij;
            }
        }
    };
    return mat_s;
}

Matrix Matrix::operator+(const Vector &vec) const {
    if (m_n != (int) vec.getN()) {
        throw std::logic_error("Invalid addition Mat+Vec");
    }

    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] + vec(i);
            }
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid substraction Mat-Mat");
    }

    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] - mat(i,j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid addition Mat+Mat");
    }
    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result, mat) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] + mat(i,j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator+(const double coeff) const {
    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] + coeff;
            }
        }
    }
    return result;
}

Matrix Matrix::operator/(const Matrix& mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid division Mat/Mat");
    }

    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] / mat(i,j);
            }
        }
    }
    return result;
}


Matrix Matrix::operator/(const double coeff) const {
    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j]/coeff;
            }
        }
    }
    return result;
}

double Matrix::sumElem() const {
    double sum = 0;
    #pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < m_m; j++) {
            sum += m_coefficients[i * m_m + j];
        }
    }
    return sum;
}

Matrix::Matrix(const Matrix& mat) {
    m_n = mat.getN();
    m_m = mat.getM();
    m_coefficients = new double[m_n*m_m];
    std::copy(mat.m_coefficients, mat.m_coefficients + m_n*m_m, m_coefficients);
}

Matrix& Matrix::operator=(const Matrix& mat) {
    if (mat.getN() != m_n || mat.getM() != m_m) {
        delete[] m_coefficients;
        m_n = mat.getN();
        m_m = mat.getM();
        m_coefficients = new double[m_n * m_m];
    }
    std::copy(mat.m_coefficients, mat.m_coefficients + m_n*m_m, m_coefficients);
    return *this;
}

Matrix Matrix::hadamardProduct(const Matrix &mat) const {
    if (m_n != mat.getN() || m_m != mat.getM()) {
        throw std::logic_error("Invalid Matrix Hadamard Product");
    }

    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                result(i, j) = m_coefficients[i * m_m + j] * mat(i, j);
            }
        }
    }
    return result;
}

Matrix Matrix::sqrt() const {
    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                if (m_coefficients[i* m_m +j] < 0) {
                    throw std::logic_error("Invalid Matrix sqrt");
                }
                result(i, j) = std::sqrt(m_coefficients[i * m_m + j]);
            }
        }
    }
    return result;
}

Matrix Matrix::log() const {
    Matrix result(m_n, m_m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_n; i++) {
            for (j = 0; j < m_m; j++) {
                if (m_coefficients[i* m_m +j] <= 0) {
                    throw std::logic_error("Invalid Matrix log");
                }
                result(i, j) = std::log(m_coefficients[i * m_m + j]);
            }
        }
    }
    return result;
}

Matrix Matrix::argmax() const {
    Matrix result(m_n, m_m);
    int i, j;
    double max_val;
    double max_i;
    #pragma omp parallel shared(result) private(i, j, max_val, max_i)
    {
        #pragma omp for
        for (j = 0; j < m_m; j++) {
            max_val = m_coefficients[j];
            max_i = 0;
            result(0, j) = 1;
            for (i = 1; i < m_n; i++) {
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
    }
    return result;
}

Matrix Matrix::generateBitMatrix(int n, int m, double bit_rate) {
    Matrix result(n, m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        unsigned int seed = omp_get_thread_num() * time(NULL);
        #pragma omp for collapse(2)
        for (i = 0; i<n; i++) {
            for (j =0; j<m; j++) {
                double r = ((double) rand_r(&seed) / (double) RAND_MAX);
                if (r < bit_rate) {
                    result(i, j) = 1;
                } else {
                    result(i, j) = 0;
                }
            }
        }
    }
    return result;
}

void Matrix::fillRandomly() {
    int i;
    double weights_init;
    #pragma omp parallel private(i, weights_init)
    {
        weights_init = std::sqrt(12.0 / (m_n + m_m));;
        unsigned int seed = omp_get_thread_num() * time(NULL);
        #pragma omp for
        for (i = 0; i < m_m * m_n; i++) {
            double r = ((double) rand_r(&seed) / (double) RAND_MAX) * 2 * weights_init - weights_init;
            m_coefficients[i] = r;
        }
    }
}

void Matrix::fillWithZero() {
    int i;
    #pragma omp parallel private(i)
    {
        #pragma omp for
        for (i = 0; i < m_m * m_n; i++) {
            m_coefficients[i] = 0;
        }
    }
}

Matrix Matrix::transpose() const { //cache aware
    Matrix transpose(m_m, m_n);
    int blocksize = 16;
    int i,j, row, col;
    #pragma omp parallel shared(transpose) private(i, j, row, col)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < m_m; i += blocksize) {
            for (j = 0; j < m_n; j += blocksize) {
                for (row = i; row < i + blocksize && row < m_m; row++) {
                    for (col = j; col < j + blocksize && col < m_n; col++) {
                        transpose(row, col) = m_coefficients[col * m_m + row];
                    }
                }
            }
        }
    }
    return transpose;
}

void Matrix::resize(int new_n, int new_m) {
    //m_coefficients = std::vector<double>(new_n * new_m);
    delete[] m_coefficients;
    m_coefficients = new double[new_n * new_m];
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
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                result(i, j) = coeff * mat(i,j);
            }
        }
    }
    return result;
}

Matrix operator-(const double coeff, const Matrix& mat) {
    int n = mat.getN();
    int m = mat.getM();
    Matrix result(n, m);
    int i, j;
    #pragma omp parallel shared(result) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < n; i++) {
            for (j = 0; j < m; j++) {
                result(i, j) = coeff - mat(i,j);
            }
        }
    }
    return result;
}
