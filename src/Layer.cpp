#include "Layer.h"

#include <iostream>
#include "Utils.h"

Layer::~Layer() {
    delete m_weights;
}

Layer::Layer(int input_dim, int neurons_number, std::string function_name) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number)
{
    m_weights = new Matrix(neurons_number, input_dim);
    m_weights->fillRandomly();
    m_bias = std::vector<double>();
    for (int i = 0; i < m_neurons_number; i++) {
        double r = ((double) rand() / (double) RAND_MAX);
        m_bias.push_back(r);
    }
    if (function_name.compare("sigmoid") == 0) {
        m_f = new  SigmoidFunction();
    }
}

void Layer::setWeights(Matrix weights) {
    //*m_weights = weights;
}

void Layer::setBias(std::vector<double> bias) {
    m_bias = bias;
}

std::vector<double> Layer::activate(const std::vector<double>& x) {
    // std::cout << m_weights << std::endl;
    // for (int i = 0; i < input.size(); i++) {
    //     std::cout << input[i] << " ";
    // }
    // std::cout << std::endl;
    // std::vector<double> e = m_weights * input;
    // std::vector<double> fe = m_f->eval(e);
    // for (int i = 0; i < e.size(); i++) {
    //     std::cout << e[i] << " ";
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < fe.size(); i++) {
    //     std::cout << fe[i] << " ";
    // }
    return m_f->eval(x);
}

std::vector<double> Layer::multiply(const std::vector<double>& input) {
    return (*m_weights) * input;
}


std::vector<double> Layer::add(const std::vector<double>& v) {
    return v + m_bias;
}

void Layer::updateWeights(const std::vector<double>& a, const std::vector<double>& delta, double learning_rate) {
    int M = m_weights->getM();
    int N = m_weights->getN();
    for (int i = 0 ; i<N ; i++) {
        for (int j = 0 ; j<M ; j++) {
            (*m_weights)(i,j) -= learning_rate * a[j] * delta[i];
        }
    }

}

void Layer::updateBias(const std::vector<double>& delta, double learning_rate) {
    int N = delta.size();
    for (int i = 0 ; i<N ; i++) {
        m_bias[i] -= learning_rate * delta[i];
    }

}


std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << *layer.getWeights();
    out << "And bias is: " << layer.getBias();
    return out;
}
