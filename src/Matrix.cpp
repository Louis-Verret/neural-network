#include "Matrix.h"

#include <cstdio>
#include <cstdlib>
#include <time.h>

Matrix::Matrix() {

}

Matrix::~Matrix() {

}

Matrix::Matrix(int input_dim, int output_dim) :
        m_n(input_dim),
        m_m(output_dim) {
    srand(time(NULL));
    m_coefficients = std::vector<double>();
    for (int i = 0; i < output_dim * input_dim; i++) {
        m_coefficients.push_back(((double) rand() / (double) RAND_MAX));
    }
}

Matrix::~Matrix() {
}

const double &Matrix::operator()(int i, int j) const {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    perror("Not valid element");
}

double &Matrix::operator()(int i, int j) {
    if (i < m_n && j < m_m)
        return m_coefficients[i * m_m + j];
    perror("Not valid element");
}

std::vector<double> Matrix::operator*(const std::vector<double> &vec) const {
    if (vec.size() != m_m)
        perror("Not valid multiplication");

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

std::ostream& operator << (std::ostream& out, const Matrix& mat) {
    int n = mat.getN(); int m = mat.getM();
    out << "Matrice "  << n << "*" << m << "  : " << std::endl ;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out << mat(i,j) << " ";
        }
        out << std::endl;
    }
    return out;
}