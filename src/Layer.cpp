#include "Layer.h"

#include <iostream>
#include <cstring>
#include "Utils.h"

Layer::~Layer() {
    delete m_f;
}

Layer::Layer(int input_dim, int neurons_number, char const* function_name) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number),
 m_weights(neurons_number, input_dim),
 m_V_dW(neurons_number, input_dim),
 m_S_dW(neurons_number, input_dim)
{
    m_weights.fillRandomly();
    m_V_dW.fillWithZero();
    m_S_dW.fillWithZero();
    m_bias = std::vector<double>();
    m_V_dB = std::vector<double>();
    m_S_dB = std::vector<double>();
    double weights_init = 4 * std::sqrt(6 / (neurons_number + input_dim));
    if (strcmp(function_name, "sigmoid") == 0) {
        m_f = new SigmoidFunction();
        //weights_init = std::sqrt(6 / (neurons_number + input_dim));
    } else if (strcmp(function_name, "tanh") == 0) {
        m_f = new TanhFunction();
        //weights_init = 4 * std::sqrt(6 / (neurons_number + input_dim));
    } else if (strcmp(function_name, "relu") == 0) {
        m_f = new ReLUFunction();
        //weights_init = std::sqrt(12 / (neurons_number + input_dim));
    }
    for (int i = 0; i < m_neurons_number; i++) {
        double r = ((double) rand() / (double) RAND_MAX) * 2 * weights_init - weights_init;
        m_bias.push_back(r);
        m_V_dB.push_back(0);
        m_S_dB.push_back(0);
    }
}

Matrix Layer::multiply(const Matrix& input) {
    return m_weights * input;
}

Matrix Layer::add(Matrix v) {
    return v + m_bias;
}

Matrix Layer::activate(const Matrix& x) {
    return m_f->eval(x);
}

std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << layer.getWeights();
    out << "And bias is: " << layer.getBias();
    return out;
}
