#ifndef LAYER
#define LAYER

#include <vector>

#include "Matrix.h"

class Layer
{
public:
    Layer(int input_dim, int neurons_number);
    ~Layer();

    int getNeuronsNumber() const {return m_neurons_number;};
    int getInputDim() const {return m_input_dim;};
    Matrix getWeights() const {return m_weights;};

    std::vector<double> computeOutput(const std::vector<double>& input);

protected:
    Matrix m_weights;
    int m_input_dim;
    int m_neurons_number;
    //ActivationFunction m_f;
    std::vector<double> m_bias;
};

#endif
