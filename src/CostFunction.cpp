#include "CostFunction.h"
#include <iostream>

/* Constructors */

CostFunction::CostFunction()
{

}

CostFunction::~CostFunction()
{

}

MSE::MSE()
{
    m_name = "mse";
}

MSE::~MSE()
{

}

CrossEntropy::CrossEntropy()
{
    m_name = "cross_entropy";
}

CrossEntropy::~CrossEntropy()
{

}


/* Methods */

Matrix MSE::computeError(const Matrix& a, const Matrix& y) const {
    Matrix diff = a - y;
    return 0.5 * diff.hadamardProduct(diff);
}

Matrix MSE::computeErrorGradient(const Matrix& a, const Matrix& y) const {
    return a - y;
}


Matrix CrossEntropy::computeError(const Matrix& a, const Matrix& y) const {
    return -1 * y.hadamardProduct(a.log());
}

Matrix CrossEntropy::computeErrorGradient(const Matrix& a, const Matrix& y) const {
    return -1 * (y / a);
}
