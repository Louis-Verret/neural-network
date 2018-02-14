#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include "Matrix.h"

/** Miscellaneous methods **/

/* Methods that reads a CSV file
   The label must be at the end of each line
   */
void readCSV(const char* file_name, bool header, Matrix& x, Matrix& y);

/* Generate normalized sinus data between -pi and pi */
void generateSinusData(Matrix& x, Matrix& d, int s);

/* Statistical methods for neural network */
void normalizeData(Matrix& x);
void centralizeData(Matrix& x, double min_value = 0, double max_value = 0);
void oneHotEncoding(Matrix& y, int n_class);

#endif
