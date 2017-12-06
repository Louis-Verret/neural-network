#include "Layer.h"

Layer::~Layer() {
}

Layer::Layer(int input_dim, int neurons_number) :
 m_input_dim(input_dim),
 m_neurons_number(neurons_number)
{
}

std::vector<double> Layer::computeOutput(const std::vector<double>& input) {
    return m_weights * input;
}
