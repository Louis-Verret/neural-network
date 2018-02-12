#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <vector>
#include <cmath>
#include "Matrix.h"

/** Virtual class used for contructing different activation functions for training a neural network.
    This class provides two virtual methods: one for evaluating the function and
    another for its derivative
**/
class ActivationFunction
{
public:

    ActivationFunction();
    virtual ~ActivationFunction();

    virtual Matrix eval(const Matrix& z) const = 0;
    virtual Matrix evalDev(const Matrix& z) const = 0;

    const char* getName() const {return m_name;};

protected:
    const char* m_name;
};


/** Class that implements the linear activation function
**/
class LinearFunction : public  ActivationFunction
{
public:

    LinearFunction();
    ~LinearFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};


/** Class that implements the sigmoid activation function
**/
class SigmoidFunction : public  ActivationFunction
{
public:

    SigmoidFunction();
    ~SigmoidFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};


/** Class that implements the tanh activation function
**/
class TanhFunction : public  ActivationFunction
{
public:

    TanhFunction();
    ~TanhFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};


/** Class that implements the Rectifier Linear Unit (ReLU) activation function
**/
class ReLUFunction : public  ActivationFunction
{
public:

    ReLUFunction();
    ~ReLUFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};


/** Class that implements the softmax activation function 
**/
class SoftmaxFunction : public  ActivationFunction
{
public:

    SoftmaxFunction();
    ~SoftmaxFunction();

    virtual Matrix eval(const Matrix& z) const;
    virtual Matrix evalDev(const Matrix& z) const;
};

#endif // ACTIVATION_FUNCTION_H
