#include "Utils.h"

#include <fstream>
#include <cmath>

void readCSV(const char* file_name, bool header, Matrix& x, Matrix& y) {
    std::ifstream file(file_name);
    std::string value;
    std::vector<std::vector<double> > read_x;
    std::vector<std::vector<double> > read_y;
    if (header) {
        getline(file, value);
    }
    while(!getline(file, value).eof()) {
        int beg = -1;
        std::vector<double> read_xi;
        std::vector<double> read_yi;
        for (unsigned int end = 0; end<value.length(); end++) {
            if (value[end] == ',' && beg != -1) {
                //std::cout << value.substr(beg+1, end-beg-1) << std::endl;
                double value_double_x = std::stod(value.substr(beg+1, end-beg-1));
                read_xi.push_back(value_double_x);
                beg = end;
            } else if (value[end] == ',' && beg == -1) {
                double value_double_y = std::stod(value.substr(beg+1, end-beg-1));
                read_yi.push_back(value_double_y);
                beg = end;
            } else if (end == value.length()-1) {
                //std::cout << value.substr(beg+1, end-beg) << std::endl;
                double value_double_x = std::stod(value.substr(beg+1, end-beg));
                read_xi.push_back(value_double_x);
                beg = end;
            }
        }
        read_x.push_back(read_xi);
        read_y.push_back(read_yi);
    }
    x.resize(read_x[0].size(), read_x.size());
    y.resize(read_y[0].size(), read_y.size());
    for(int i = 0; i<x.getN(); i++) {
        for(int j = 0; j<x.getM(); j++) {
            x(i, j) = read_x[j][i]/255;
        }
    }
    for(int i = 0; i<y.getN(); i++) {
        for(int j = 0; j<y.getM(); j++) {
            y(i, j) = read_y[j][i];
        }
    }

}

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

void oneHotEncoding(Matrix& y, int n_class) {
    int m = y.getM();
    Matrix y_copy = y;
    y.resize(n_class, m);
    for (int j = 0; j < m; j++) {
        int val = y_copy(0, j);
        for (int i = 0; i < n_class; i++) {
            if (i == val) {
                y(i, j) = 1;
            } else {
                y(i, j) = 0;
            }
        }
    }
}
