#include "Utils.h"

#include <cmath>

void generateSinusData(Matrix& x, Matrix& y, int s) {
    srand(time(NULL));
    int lower_bound_x = -4, upper_bound_x = 4;
    int lower_bound_y = -1, upper_bound_y = 1;
    x.resize(1, s);
    y.resize(1, s);
    for (int i = 0; i<s; i++) {
        double input = ((double)rand() / (double)RAND_MAX) * 6.28 - 3.14;
        x(0, i) = input;
        y(0, i) = sin(input);
    }
    for (int i = 0; i<s; i++) {
        x(0, i) = (x(0, i) - lower_bound_x) / (upper_bound_x - lower_bound_x);
        y(0, i) = (y(0, i) - lower_bound_y) / (upper_bound_y - lower_bound_y);
    }
}
