#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "Matrix.h"

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator*(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator/(const std::vector<double>& v1, const std::vector<double>& v2);
std::vector<double> operator*(const double coeff, const std::vector<double>& v);
std::vector<double> operator+(const std::vector<double>& v, const double coeff);
std::vector<double> operator/(const std::vector<double>& v, const double coeff);
std::vector<double> sqrt(const std::vector<double>& v);
std::ostream& operator << (std::ostream& out, const std::vector<double>& v);

void readCSV(const char* file_name, Matrix& x, Matrix& y);
void generateSinusData(Matrix& x, Matrix& d, int s);

#endif
