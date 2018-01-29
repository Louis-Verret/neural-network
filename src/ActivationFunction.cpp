#include "ActivationFunction.h"
#include <iostream>

ActivationFunction::ActivationFunction()
{

}

ActivationFunction::~ActivationFunction()
{

}

LinearFunction::LinearFunction()
{

}

LinearFunction::~LinearFunction()
{

}

SigmoidFunction::SigmoidFunction()
{
    m_name = "sigmoid";
}

SigmoidFunction::~SigmoidFunction()
{

}

TanhFunction::TanhFunction()
{
    m_name = "tanh";
}

TanhFunction::~TanhFunction()
{

}

ReLUFunction::ReLUFunction()
{
    m_name = "relu";
}

ReLUFunction::~ReLUFunction()
{

}

SoftmaxFunction::SoftmaxFunction()
{
    m_name = "softmax";
}

SoftmaxFunction::~SoftmaxFunction()
{

}

Matrix LinearFunction::eval(const Matrix& z) const {
    return z;
}

Matrix LinearFunction::evalDev(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         result(i, j) = 1;
    //     }
    // }
    // return result;
    Matrix result = z.computeLinearDev();
}

Matrix SigmoidFunction::eval(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         result(i, j) = 1.0 / (1.0 + exp(-z(i, j)));
    //     }
    // }
    // return result;
    return z.computeSigmoidEval();
}

Matrix SigmoidFunction::evalDev(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         double eval = 1.0 / (1.0 + exp(-z(i, j)));
    //         result(i, j) = (eval) * (1 - eval);
    //     }
    // }
    // return result;
    return z.computeSigmoidDev();
}

Matrix TanhFunction::eval(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         result(i, j) = std::tanh(z(i, j));
    //     }
    // }
    // return result;
    return z.computeTanhEval();
}

Matrix TanhFunction::evalDev(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         double eval =  std::tanh(z(i, j));
    //         result(i, j) = 1 - std::pow(eval, 2);
    //     }
    // }
    // return result;
    return z.computeTanhDev();
}

Matrix ReLUFunction::eval(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         result(i, j) = std::max(0.0, z(i, j));
    //     }
    // }
    // return result;
    return z.computeReLUEval();
}

Matrix ReLUFunction::evalDev(const Matrix& z) const {
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         if (z(i, j) > 0) {
    //             result(i, j) = 1;
    //         } else {
    //             result(i, j) = 0;
    //         }
    //     }
    // }
    // return result;
    return z.computeReLUDev();
}


Matrix SoftmaxFunction::eval(const Matrix& z) const {
    // std::cout << z << std::endl;
    // int n = z.getN();
    // int m = z.getM();
    // Matrix result(n, m);
    // Vector sum_expo(m, 0);
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         sum_expo(j) += std::exp(z(i, j));
    //     }
    // }
    // for (int i = 0; i<n ; i++) {
    //     for (int j = 0; j<m ; j++) {
    //         result(i, j) = std::exp(z(i, j)) / sum_expo(j);
    //     }
    // }
    // return result;
    return z.computeSoftmaxEval();
}

Matrix SoftmaxFunction::evalDev(const Matrix& z) const {
}
