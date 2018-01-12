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
    int n = a.getN();
    int m = a.getM();
    int n_errors = 0;
    Matrix arg_max_a = a.argmax();
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
             if (arg_max_a(i, j) != y(i, j)) {
                 n_errors++;
                 break;
             }
        }
    }
    return 100 - n_errors * 100 / m;
}
