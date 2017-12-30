#include "Layer.h"

#include <iostream>
#include <cstring>
#include "Utils.h"

Layer::~Layer() {
    delete m_weights;
    delete m_V_dW;
    delete m_S_dW;
    delete m_f;
}

Layer::Layer(int input_dim, int neurons_number, char const* function_name) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number)//,
 //m_weights(neurons_number, input_dim)
{
    m_weights = new Matrix(neurons_number, input_dim);
    m_V_dW = new Matrix(neurons_number, input_dim);
    m_S_dW = new Matrix(neurons_number, input_dim);
    m_weights->fillRandomly();
    m_V_dW->fillWithZero();
    m_S_dW->fillWithZero();
    m_bias = std::vector<double>();
    m_V_dB = std::vector<double>();
    m_S_dB = std::vector<double>();
    for (int i = 0; i < m_neurons_number; i++) {
        double xavier_init = 4 * std::sqrt(6 / (neurons_number + input_dim));
        double r = ((double) rand() / (double) RAND_MAX) * 2 * xavier_init - xavier_init;
        m_bias.push_back(r);
        m_V_dB.push_back(0);
        m_S_dB.push_back(0);
    }
    if (strcmp(function_name, "sigmoid") == 0) {
        m_f = new SigmoidFunction();
    } else if (strcmp(function_name, "tanh") == 0) {
        m_f = new TanhFunction();
    }
}

Matrix Layer::multiply(const Matrix& input) {
    return (*m_weights) * input;
}

Matrix Layer::add(Matrix v) {
    return v + m_bias;
}

Matrix Layer::activate(const Matrix& x) {
    return m_f->eval(x);
}

// void  Layer::updateWeights(const Matrix& a, const Matrix& delta, double learning_rate, int batch_size, double momentum) {
//     Matrix dW = (delta * a.transpose()) / batch_size;
//     Matrix update_mat = dW + momentum * (*getLastWeights());
//     Matrix new_weights = (*getWeights()) - learning_rate * update_mat;
//     setWeights(new_weights);
//     setLastWeights(update_mat);
//
//     // int M = m_weights->getM();
//     // int N = m_weights->getN();
//     // for (int i = 0 ; i<N ; i++) {
//     //     for (int j = 0 ; j<M ; j++) {
//     //         double update_val = (dW(i, j)/batch_size) + momentum * (*m_V_dW)(i, j);
//     //         (*m_weights)(i, j) -= learning_rate * update_val;
//     //         (*m_V_dW)(i, j) = update_val;
//     //     }
//     // }
// }
//
// void Layer::updateBias(const Matrix& delta, double learning_rate, int batch_size, double momentum) {
//     std::vector<double> ones(delta.getM(), 1);
//     std::vector<double> dB = (delta * ones) / batch_size;
//     std::vector<double> update_vec = dB + momentum * getLastBias();
//     std::vector<double> new_bias = getBias() - learning_rate * update_vec;
//     setBias(new_bias);
//     setLastBias(update_vec);
//
//     // int N = delta.getN();
//     // int M = delta.getM();
//     // for (int i = 0 ; i<N ; i++) {
//     //     double sum = 0;
//     //     for (int j = 0; j<M; j++) {
//     //         sum += delta(i, j);
//     //     }
//     //     double update_val =  (sum/batch_size) + momentum * m_V_dB[i];
//     //     m_bias[i] -= learning_rate * update_val;
//     //     m_V_dB[i] = update_val;
//     // }
// }

std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << *layer.getWeights();
    out << "And bias is: " << layer.getBias();
    return out;
}
