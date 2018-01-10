#include "CostFunction.h"

CostFunction::CostFunction()
{

}

CostFunction::~CostFunction()
{

}

MSE::MSE()
{

}

MSE::~MSE()
{

}

Matrix MSE::computeError(const Matrix& a, const Matrix& y) const {
    Matrix diff = a - y;
    return 0.5 * diff.hadamardProduct(diff);
}

Matrix MSE::computeErrorGradient(const Matrix& a, const Matrix& y) const {
    return a - y;
}

CrossEntropy::CrossEntropy()
{

}

CrossEntropy::~CrossEntropy()
{

}


Matrix CrossEntropy::computeError(const Matrix& a, const Matrix& y) const {
    return -1 * y.hadamardProduct(a.log());
}

Matrix CrossEntropy::computeErrorGradient(const Matrix& a, const Matrix& y) const {
    return -1 * (y / a);
}
