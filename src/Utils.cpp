#include "Utils.h"

#include <cmath>

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        perror("Invalid size for vector addition");
    }
    std::vector<double> res;
    for (int i = 0; i < (int)v1.size(); i++) {
        res.push_back(v1[i] + v2[i]);
    }
    return res;
}

std::vector<double> operator*(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        perror("Invalid size for Hadamard product");
    }
    std::vector<double> res;
    for (int i = 0; i < (int)v1.size(); i++) {
        res.push_back(v1[i] * v2[i]);
    }
    return res;
}

std::ostream& operator << (std::ostream& out, const std::vector<double>& v) {
    int n = v.size();
    out << "( ";
    for (int i = 0; i < n; i++) {
        out << v[i] << " ";
    }
    out << ")";
    return out;
}

void generateSinusData(Matrix& x, Matrix& d, int s) {
    srand(time(NULL));
    int lower_bound_x = -1, upper_bound_x = 4;
    int lower_bound_d = -1, upper_bound_d = 1;
    x.resize(1, s);
    d.resize(1, s);
    for (int i = 0; i<s; i++) {
        double input = ((double)rand() / (double)RAND_MAX) * 6.28 - 3.14;
        x(0, i) = input;
        d(0, i) = sin(input);
    }
    for (int i = 0; i<s; i++) {
        x(0, i) = (x(0, i) - lower_bound_x) / (upper_bound_x - lower_bound_x);
        d(0, i) = (d(0, i) - lower_bound_d) / (upper_bound_d - lower_bound_d);
    }
}
