#ifndef MATRIX
#define MATRIX

#include <vector>

class Matrix
{
public:
    Matrix();
    Matrix(int n, int m, std::vector<double> coefficients);
    ~Matrix();
    int getN() const {return m_n;};
    int getM() const {return m_m;};

    double& operator () (int i, int j);

protected:
    int m_n;
    int m_m;
    std::vector<double> m_coefficients;

};

#endif
