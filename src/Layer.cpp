#include "Layer.h"

#include <iostream>
#include "Utils.h"

Layer::~Layer() {
    delete m_weights;
    delete m_last_update_weights;
    delete m_f;
}

Layer::Layer(int input_dim, int neurons_number, std::string function_name) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number)//,
 //m_weights(neurons_number, input_dim)
{
    m_weights = new Matrix(neurons_number, input_dim);
    m_last_update_weights = new Matrix(neurons_number, input_dim);
    m_weights->fillRandomly();
    m_last_update_weights->fillWithZero();
    m_bias = std::vector<double>();
    m_last_update_bias = std::vector<double>();
    for (int i = 0; i < m_neurons_number; i++) {
        double r = ((double) rand() / (double) RAND_MAX);
        m_bias.push_back(r);
        m_last_update_bias.push_back(0);
    }
    if (function_name.compare("sigmoid") == 0) {
        m_f = new  SigmoidFunction();
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

void  Layer::updateWeights(const Matrix& a, const Matrix& delta, double learning_rate, int mini_batch, double momentum) {
    int M = m_weights->getM();
    int N = m_weights->getN();
    Matrix calc = delta * a.transpose();
    for (int i = 0 ; i<N ; i++) {
        for (int j = 0 ; j<M ; j++) {
            double update_val = -(learning_rate/mini_batch) * calc(i, j) + momentum * (*m_last_update_weights)(i, j);
            (*m_weights)(i, j) += update_val;
            (*m_last_update_weights)(i, j) = update_val;
        }
    }
}

void Layer::updateBias(const Matrix& delta, double learning_rate, int mini_batch, double momentum) {
    int N = delta.getN();
    int M = delta.getM();
    for (int i = 0 ; i<N ; i++) {
        double sum = 0;
        for (int j = 0; j<M; j++) {
            sum += delta(i, j);
        }
        double update_val = -(learning_rate/mini_batch) * sum + momentum * m_last_update_bias[i];
        m_bias[i] += update_val;
        m_last_update_bias[i] = update_val;
    }

}

std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << *layer.getWeights();
    out << "And bias is: " << layer.getBias();
    return out;
}
