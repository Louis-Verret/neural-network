#include "ActivationFunction.h"

ActivationFunction::ActivationFunction()
{

}

ActivationFunction::~ActivationFunction()
{

}

SigmoidFunction::SigmoidFunction()
{

}

SigmoidFunction::~SigmoidFunction()
{

}

Matrix SigmoidFunction::eval(const Matrix& z) const {
    int n = z.getN();
    int m = z.getM();
    Matrix result(n, m);
    for (int i = 0; i<n ; i++) {
        for (int j = 0; j<m ; j++) {
            result(i, j) = 1.0 / (1.0 + exp(-z(i, j)));
        }
    }
    return result;
}

Matrix SigmoidFunction::evalDev(const Matrix& z) const {
    int n = z.getN();
    int m = z.getM();
    Matrix result(n, m);
    for (int i = 0; i<n ; i++) {
        for (int j = 0; j<m ; j++) {
            double eval = 1.0 / (1.0 + exp(-z(i, j)));
            result(i, j) = (eval) * (1 - eval);
        }
    }
    return result;
}
