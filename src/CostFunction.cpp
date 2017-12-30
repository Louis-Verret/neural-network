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
    int K = a.getN();
    Matrix diff = a - y;
    return (0.5/K) * diff.hadamardProduct(diff);
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
    int K = a.getN();
    Matrix diff = y.hadamardProduct(a.log()) + (1 - y).hadamardProduct((1 - a).log());
    return (1/K) * diff;
}

Matrix CrossEntropy::computeErrorGradient(const Matrix& a, const Matrix& y) const {
    return (a - y) / (a.hadamardProduct(1 - a));
}
