#include "Utils.h"

#include <stdexcept>
#include <fstream>
#include <cmath>

void readCSV(const char* file_name, bool header, Matrix& x, Matrix& y) {
    std::ifstream file(file_name);
    std::string value;
    std::vector<std::vector<double> > read_x;
    std::vector<std::vector<double> > read_y;
    if (file.is_open()) {
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
        int blocksize = 16;
        int i, j, row, col;
        #pragma omp parallel shared(x, read_x, blocksize) private(i, j, row, col)
        {
            #pragma omp for
            for (i = 0; i < x.getN(); i += blocksize) {
                for (j = 0; j < x.getM(); j += blocksize) {
                    for (row = i; row < i + blocksize && row < x.getN(); row++) {
                        for (col = j; col < j + blocksize && col < x.getM(); col++) {
                            x(row, col) = read_x[col][row];
                        }
                    }
                }
            }
        }
        #pragma omp parallel shared(y, read_y, blocksize) private(i, j, row, col)
        {
            #pragma omp for
            for (i = 0; i < y.getN(); i += blocksize) {
                for (j = 0; j < y.getM(); j += blocksize) {
                    for (row = i; row < i + blocksize && row < y.getN(); row++) {
                        for (col = j; col < j + blocksize && col < y.getM(); col++) {
                            y(row, col) = read_y[col][row];
                        }
                    }
                }
            }
        }
    } else {
        throw std::logic_error("Can't open file.");
    }
}

void normalizeData(Matrix& x) {
    double mean = 0;
    double sd = 0;
    int i, j;
    #pragma omp parallel for reduction (+:mean)
    for (i = 0; i < x.getN(); i++) {
        for (j = 0; j < x.getM(); j++) {
            mean += x(i, j);
        }
    }
    mean /= (x.getN() * x.getM());
    #pragma omp parallel shared(x, mean, sd) private(i, j)
    {
        #pragma omp for
        for (i = 0; i < x.getN(); i++) {
            for (j = 0; j < x.getM(); j++) {
                sd +=  std::pow(x(i,j) - mean, 2);
            }
        }
    }
    sd = std::sqrt(sd);
    #pragma omp parallel shared(x, mean, sd) private(i, j)
    {
        #pragma omp for
        for (i = 0; i < x.getN(); i++) {
            for (j = 0; j < x.getM(); j++) {
                x(i, j) =  (x(i,j) - mean) / sd;
            }
        }
    }
}

void centralizeData(Matrix& x, double min_value, double max_value) {
    if (min_value == 0 && max_value == 0) {
        min_value = x(0, 0);
        max_value = x(0, 0);
        for (int i = 0; i < x.getN(); i++) {
            for (int j = 0; j < x.getM(); j++) {
                max_value = std::max(max_value, x(i, j));
                min_value = std::min(min_value, x(i, j));
            }
        }
    }
    int i, j;
    #pragma omp parallel shared(x, min_value, max_value) private(i, j)
    {
        #pragma omp for collapse(2)
        for (i = 0; i < x.getN(); i++) {
            for (j = 0; j < x.getM(); j++) {
                x(i, j) = (x(i, j) - min_value) / (max_value - min_value);
            }
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
