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

std::vector<double> SigmoidFunction::Eval(std::vector<double> z) const {
    std::vector<double> result;
    int n = z.size();
    for (int i = 0; i<n ; i++) {
        result.push_back(1.0 / 1.0 + exp(-z[i]));
    }
    return result;
}

std::vector<double> SigmoidFunction::EvalDev(std::vector<double> z) const {
    std::vector<double> eval = Eval(z);
    std::vector<double> result;
    int n = z.size();
    for (int i = 0; i<n ; i++) {
        result.push_back(eval[i] * (1 - eval[i]));
    }
    return result;
}
