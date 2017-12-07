#include "Layer.h"

#include <iostream>

Layer::~Layer() {
    delete m_weights;
}

Layer::Layer(int input_dim, int neurons_number, std::string function_name) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number)//,
 //m_weights(neurons_number, input_dim)
{
    m_weights = new Matrix(neurons_number, input_dim);
    if (function_name.compare("sigmoid") == 0) {
        m_f = new  SigmoidFunction();
    }
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

void Layer::updateWeights(const std::vector<double>& a, const std::vector<double>& delta, double learning_rate) {
    int M = m_weights->getM();
    int N = m_weights->getN();
    for (int i = 0 ; i<N ; i++) {
        for (int j = 0 ; j<M ; j++) {
            (*m_weights)(i,j) -= learning_rate * a[j] * delta[i];
        }
    }

}


std::ostream& operator << (std::ostream& out, const Layer& layer) {
    out << "Layer with " << layer.getInputDim() << " input dimension(s) and " << layer.getNeuronsNumber() << " neuron(s)." << std::endl;
    out << "Costs are:" << std::endl;
    out << layer.getWeights();
    return out;
}
