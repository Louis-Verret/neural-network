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

std::vector<double> SigmoidFunction::eval(std::vector<double> z) const {
    std::vector<double> result;
    int n = z.size();
    for (int i = 0; i<n ; i++) {
        result.push_back(1.0 / (1.0 + exp(-z[i])));
    }
    return result;
}

std::vector<double> SigmoidFunction::evalDev(std::vector<double> z) const {
    std::vector<double> val = eval(z);
    std::vector<double> result;
    int n = z.size();
    for (int i = 0; i<n ; i++) {
        result.push_back(val[i] * (1 - val[i]));
    }
    return result;
}
