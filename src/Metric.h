#ifndef METRIC_H
#define METRIC_H

#include "Matrix.h"

class Metric
{
public:

    Metric();
    virtual ~Metric();

    virtual double computeMetric(const Matrix& a, const Matrix& y) const = 0;
};

class CategoricalAccuracy : public Metric
{
public:

    CategoricalAccuracy();
    ~CategoricalAccuracy();

    virtual double computeMetric(const Matrix& a, const Matrix& y) const;
};

#endif
