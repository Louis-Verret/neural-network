#include "ActivationFunction.h"
#include <iostream>

/* Constructors */

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

/* Activation Function Methods (eval and evalDev) */

Matrix LinearFunction::eval(const Matrix& z) const {
    return z;
}

Matrix LinearFunction::evalDev(const Matrix& z) const {
    return z.computeLinearDev();
}

Matrix SigmoidFunction::eval(const Matrix& z) const {
    return z.computeSigmoidEval();
}

Matrix SigmoidFunction::evalDev(const Matrix& z) const {
    return z.computeSigmoidDev();
}

Matrix TanhFunction::eval(const Matrix& z) const {
    return z.computeTanhEval();
}

Matrix TanhFunction::evalDev(const Matrix& z) const {
    return z.computeTanhDev();
}

Matrix ReLUFunction::eval(const Matrix& z) const {
    return z.computeReLUEval();
}

Matrix ReLUFunction::evalDev(const Matrix& z) const {
    return z.computeReLUDev();
}


Matrix SoftmaxFunction::eval(const Matrix& z) const {
    return z.computeSoftmaxEval();
}

Matrix SoftmaxFunction::evalDev(const Matrix& z) const {
}
