#include "Metric.h"

#include <iostream>

Metric::Metric()
{

}

Metric::~Metric()
{

}

CategoricalAccuracy::CategoricalAccuracy()
{

}

CategoricalAccuracy::~CategoricalAccuracy()
{

}

double CategoricalAccuracy::computeMetric(const Matrix& a, const Matrix& y) const {
    return a.computeMetric(y);
}
