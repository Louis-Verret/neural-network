#include "Matrix.h"

double& Matrix::operator () (int i, int j) {

    return m_coefficients[i * m_m + j];
}
