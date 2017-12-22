#ifndef LAYER
#define LAYER

#include <vector>
#include <string>
#include "Matrix.h"
#include "ActivationFunction.h"

class Layer
{
public:
    Layer(int input_dim, int neurons_number, std::string function_name);
    ~Layer();

    int getNeuronsNumber() const {return m_neurons_number;};
    int getInputDim() const {return m_input_dim;};
    Matrix* getWeights() const {return m_weights;};
    std::vector<double> getBias() const {return m_bias;};
    ActivationFunction* getActivationFunction() const {return m_f;};

    //std::vector<double> computeOutput(const std::vector<double>& input);
    std::vector<double> multiply(const std::vector<double>& input);
    std::vector<double> add(const std::vector<double>& v);
    std::vector<double> activate(const std::vector<double>& x);

    // Matrix multiply(const Matrix& input);
    // Matrix add(const Matrix& v);
    // Matrix activate(const Matrix& x);

    void updateWeights(const std::vector<double>& a, const std::vector<double>& delta, double learning_rate);
    void updateBias(const std::vector<double>& delta, double learning_rate);


protected:
    Matrix* m_weights;
    int m_input_dim;
    int m_neurons_number;
    ActivationFunction* m_f;
    std::vector<double> m_bias;
};

std::ostream& operator << (std::ostream& out, const Layer& layer);

#endif
