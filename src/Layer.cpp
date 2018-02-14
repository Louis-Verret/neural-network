#include "Layer.h"

#include <iostream>
#include <cstring>
#include "Utils.h"

/* Constructor / Destructor */
Layer::~Layer() {
    delete m_f;
}

Layer::Layer(int input_dim, int neurons_number, char const* function_name, bool init) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number),
 m_weights(neurons_number, input_dim),
 m_bias(neurons_number),
 m_V_dW(neurons_number, input_dim),
 m_S_dW(neurons_number, input_dim),
 m_V_dB(neurons_number),
 m_S_dB(neurons_number)
{
    if (init) {
        m_weights.fillRandomly();
        m_V_dW.fillWithZero();
        m_S_dW.fillWithZero();
        m_bias.fillRandomly();
        m_V_dB.fillWithZero();
        m_S_dB.fillWithZero();
    }
    if (strcmp(function_name, "sigmoid") == 0) {
        m_f = new SigmoidFunction();
    } else if (strcmp(function_name, "tanh") == 0) {
        m_f = new TanhFunction();
    } else if (strcmp(function_name, "relu") == 0) {
        m_f = new ReLUFunction();
    } else if (strcmp(function_name, "softmax") == 0) {
        m_f = new SoftmaxFunction();
    }
}


/* Methods used during the propagation step */
Matrix Layer::multiply(const Matrix& input) {
    return m_weights * input;
}

Matrix Layer::add(Matrix v) {
    return v + m_bias;
}

Matrix Layer::activate(const Matrix& x) {
    if (m_dropout_rate != 0) {
        Matrix bit_mat = Matrix::generateBitMatrix(x.getN(), x.getM(), 1 - m_dropout_rate);
        return (m_f->eval(x).hadamardProduct(bit_mat)) / m_dropout_rate;
    } else {
        return m_f->eval(x);
    }
}


/* Extern method */
std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << layer.getWeights();
    out << "Bias is: " << layer.getBias() << std::endl;
    out << "And activation function is a " << layer.getActivationFunction()->getName() << std::endl;
    return out;
}
