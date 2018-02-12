#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "MatrixCPU.h"

/** Miscellaneous methods **/

/* Methods that reads a CSV file
   The label must be at the end of each line
   */
void readCSV(const char* file_name, bool header, MatrixCPU& x, MatrixCPU& y);

/* Generate normalized sinus data between -pi and pi */
void generateSinusData(MatrixCPU& x, MatrixCPU& d, int s);

/* Statistical methods for neural network */
void normalizeData(MatrixCPU& x);
void centralizeData(MatrixCPU& x, double min_value = 0, double max_value = 0);
void oneHotEncoding(MatrixCPU& y, int n_class);

#endif
