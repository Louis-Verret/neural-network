#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "Matrix.h"

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator*(const std::vector<double>& v1, const std::vector<double>& v2);
std::ostream& operator << (std::ostream& out, const std::vector<double>& v);

void generateSinusData(Matrix& x, Matrix& d, int s);

#endif
