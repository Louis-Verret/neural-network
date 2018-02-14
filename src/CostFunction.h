#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "Matrix.h"

/** Virtual class used for contructing different cost functions for training a neural network.
    This class provides two virtual methods for computing the error and the
    derivative error
**/
class CostFunction
{
public:

    CostFunction();
    virtual ~CostFunction();

    virtual Matrix computeError(const Matrix& a, const Matrix& y) const = 0;
    virtual Matrix computeErrorGradient(const Matrix& a, const Matrix& y) const = 0;

    const char* getName() const {return m_name;};

protected:
    const char* m_name;
};


/** Class that implements the mean squared error (MSE) cost function
**/
class MSE : public CostFunction
{
public:

    MSE();
    ~MSE();

    virtual Matrix computeError(const Matrix& a, const Matrix& y) const;
    virtual Matrix computeErrorGradient(const Matrix& a, const Matrix& y) const;
};


/** Class that implements the cross entropy cost function
**/
class CrossEntropy : public CostFunction
{
public:

    CrossEntropy();
    ~CrossEntropy();

    virtual Matrix computeError(const Matrix& a, const Matrix& y) const;
    virtual Matrix computeErrorGradient(const Matrix& a, const Matrix& y) const;
};


#endif
