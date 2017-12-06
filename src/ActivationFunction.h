#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include <cmath>

class ActivationFunction
{
public:

    ActivationFunction();
    ~ActivationFunction();

    virtual std::vector<double> eval(std::vector<double> z) const = 0;
    virtual std::vector<double> evalDev(std::vector<double> z) const = 0;

};



class SigmoidFunction : public  ActivationFunction
{
public:

    SigmoidFunction();
    ~SigmoidFunction();

    virtual std::vector<double> eval(std::vector<double> z) const;
    virtual std::vector<double> evalDev(std::vector<double> z) const;
};

#endif // ACTIVATION_FUNCTION_H
