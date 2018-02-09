#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "MatrixCPU.h"

void readCSV(const char* file_name, bool header, MatrixCPU& x, MatrixCPU& y);
void normalizeData(MatrixCPU& x);
void centralizeData(MatrixCPU& x, double min_value = 0, double max_value = 0);
void generateSinusData(MatrixCPU& x, MatrixCPU& d, int s);
void oneHotEncoding(MatrixCPU& y, int n_class);

#endif
