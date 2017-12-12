#include "Utils.h"

#include <cmath>

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        perror("Invalid size for vector addition");
    }
    std::vector<double> res;
    for (int i = 0; i<v1.size(); i++) {
        res.push_back(v1[i] + v2[i]);
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

void generateSinusData(std::vector<std::vector<double> >& x, std::vector<std::vector<double> >& d, int s) {
    srand(time(NULL));
    double max_input = -10;
    double min_input = 10;
    double max_d = -1;
    double min_d = 1;
    for (int i = 0; i<s; i++) {
        std::vector<double> xi;
        std::vector<double> di;
        double input = ((double)rand() / (double)RAND_MAX) * 6.28 - 3.14;
        max_input = std::max(max_input, input);
        min_input = std::min(min_input, input);
        max_d = std::max(max_d, sin(input));
        min_d = std::min(min_d, sin(input));
        xi.push_back(input);
        di.push_back(sin(input));
        d.push_back(di);
        x.push_back(xi);
    }
    for (int i = 0; i<s; i++) {
        x[i][0] = (x[i][0] - min_input) / (max_input - min_input);
        d[i][0] = (d[i][0] - min_d) / (max_d - min_d);
    }
}