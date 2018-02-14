#ifndef METRIC_H
#define METRIC_H

#include "Matrix.h"

/** Virtual class used for contructing different metric functions for evaluating a neural network.
    This class provides a virtual method for computing metric
**/
class Metric
{
public:

    Metric();
    virtual ~Metric();

    virtual double computeMetric(const Matrix& a, const Matrix& y) const = 0;
};


/** Class that implements the categorical accuracy metric function */
class CategoricalAccuracy : public Metric
{
public:

    CategoricalAccuracy();
    ~CategoricalAccuracy();

    virtual double computeMetric(const Matrix& a, const Matrix& y) const;
};

#endif
