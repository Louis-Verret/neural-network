#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "Matrix.h"

void readCSV(const char* file_name, bool header, Matrix& x, Matrix& y);
void generateSinusData(Matrix& x, Matrix& d, int s);

#endif
