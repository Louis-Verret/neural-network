#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "Matrix.h"

void readCSV(const char* file_name, bool header, Matrix& x, Matrix& y);
void normalizeData(Matrix& x);
void centralizeData(Matrix& x, double min_value = 0, double max_value = 0);
void generateSinusData(Matrix& x, Matrix& d, int s);
void oneHotEncoding(Matrix& y, int n_class);

#endif
