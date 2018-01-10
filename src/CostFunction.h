#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "Matrix.h"

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

class MSE : public CostFunction
{
public:

    MSE();
    ~MSE();

    virtual Matrix computeError(const Matrix& a, const Matrix& y) const;
    virtual Matrix computeErrorGradient(const Matrix& a, const Matrix& y) const;
};

class CrossEntropy : public CostFunction
{
public:

    CrossEntropy();
    ~CrossEntropy();

    virtual Matrix computeError(const Matrix& a, const Matrix& y) const;
    virtual Matrix computeErrorGradient(const Matrix& a, const Matrix& y) const;
};


#endif
