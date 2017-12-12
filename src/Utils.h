#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2);

std::ostream& operator << (std::ostream& out, const std::vector<double>& v);

void generateSinusData(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int s);

#endif
