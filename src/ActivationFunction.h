#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include <cmath>
#include "Matrix.h"

class ActivationFunction
{
public:

    ActivationFunction();
    virtual ~ActivationFunction();

    virtual Matrix eval(const Matrix& z) const = 0;
    virtual Matrix evalDev(const Matrix& z) const = 0;
};



class SigmoidFunction : public  ActivationFunction
{
public:

    SigmoidFunction();
    ~SigmoidFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};

class TanhFunction : public  ActivationFunction
{
public:

    TanhFunction();
    ~TanhFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};

class ReLUFunction : public  ActivationFunction
{
public:

    ReLUFunction();
    ~ReLUFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};

#endif // ACTIVATION_FUNCTION_H
