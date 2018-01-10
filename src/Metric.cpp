#include "Metric.h"

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

Matrix CategoricalAccuracy::computeMetric(const Matrix& a, const Matrix& y) const {
    Matrix diff = a - y;
    return 0.5 * diff.hadamardProduct(diff);
}
