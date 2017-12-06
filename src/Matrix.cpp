#include "Matrix.h"
#include <iostream>


Matrix::Matrix() : m_n(0), m_m(0) {
}

Matrix::~Matrix() {
}

Matrix::Matrix(int input_dim, int output_dim) :
        m_n(input_dim),
        m_m(output_dim) {
    m_coefficients = std::vector<double>();
    for (int i = 0; i < output_dim * input_dim; i++) {
        double r = ((double) rand() / (double) RAND_MAX);
        m_coefficients.push_back(r);
    }
}

const double &Matrix::operator()(int i, int j) const {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    perror("Invalid element");
}

double &Matrix::operator()(int i, int j) {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    perror("Invalid element");
}

std::vector<double> Matrix::operator*(const std::vector<double> &vec) const {
    if (vec.size() != m_m)
        perror("Invalid multiplication");

    std::vector<double> vec_s;
    for (int i = 0; i < m_n; i++) {
        double vec_s_i = 0;
        for (int j = 0; j < m_m; j++) {
            vec_s_i += m_coefficients[i * m_m + j] * vec[j];
        }
        vec_s.push_back(vec_s_i);
    }
    return vec_s;
}

Matrix Matrix::transpose() const{ //non cache aware + matrice initialisée à des valeurs randoms à changer
    Matrix transpose(m_m, m_n);
    for (int i = 0; i<m_m; i++) {
        for (int j =0; j<m_n; j++) {
            transpose(i, j) = m_coefficients[j * m_m + i];
        }
    }
}

std::ostream& operator << (std::ostream& out, const Matrix& mat) {
    int n = mat.getN(); int m = mat.getM();
    out << "Size ("  << n << " * " << m << ")" << std::endl ;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << mat(i,j) << " ";
        }
        out << std::endl;
    }
    return out;
}
